#include <opencv2/opencv.hpp>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <tuple>
#include <map>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <errno.h>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <time.h>
#include <syslog.h>
#include "helper_funcs.hpp"
#include "cuda_process.hpp"
#include <inttypes.h>  
using namespace std;
using namespace cv;

static bool g_use_cuda = (cuda::getCudaEnabledDeviceCount() > 0);
static ProcCUDA g_cuda;

const double MIN_SIDE_OK = 80;  
const double MAX_SIDE_OK = 160;  

struct SideInfo {
    double len[4];       
    int shortest;        
    int longest;         
};

// expects corners in TL, TR, BL, BR order
inline SideInfo getParallelogramSideInfo(const vector<Center*>& c) {
    const int e[4][2] = {{0,1}, {1,3}, {3,2}, {2,0}};

    auto L = [&](int a, int b) {
        double dx = c[a]->cx - c[b]->cx;
        double dy = c[a]->cy - c[b]->cy;
        return std::hypot(dx, dy);
    };

    SideInfo s{};
    s.len[0] = L(e[0][0], e[0][1]);  
    s.len[1] = L(e[1][0], e[1][1]);  
    s.len[2] = L(e[2][0], e[2][1]);  
    s.len[3] = L(e[3][0], e[3][1]);  

    s.shortest = 0; s.longest = 0;
    for (int i = 1; i < 4; ++i) {
        if (s.len[i] < s.len[s.shortest]) s.shortest = i;
        if (s.len[i] > s.len[s.longest])  s.longest  = i;
    }
    return s;
}

static inline uint64_t elapsed_ns(uint64_t start_ns) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    uint64_t now = (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
    return now - start_ns;
}

static inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}


// Thread-safe queue for Mat
typedef queue<Mat> MatQueue;
class FrameQueue {
    MatQueue q;
    mutex mtx;
    condition_variable cond;
public:
    void push(const Mat& frame) {
        {
            lock_guard<mutex> lk(mtx);
            q.push(frame.clone());
        }
        cond.notify_one();
    }

    // Blocks until a frame is available
    bool pop(Mat& frame) {
        unique_lock<mutex> lk(mtx);
        cond.wait(lk, [&]{ return !q.empty(); });
        frame = move(q.front());
        q.pop();
        return !frame.empty();
    }
};

// Given 4 pointers (quad), return indices {TL, TR, BR, BL} into that vector.
inline std::array<int,4> order4_indices(const std::vector<Center*>& q) {
    auto S = [&](int i){ return q[i]->cx + q[i]->cy; };    
    auto D = [&](int i){ return q[i]->cx - q[i]->cy; };     

    int tl = 0, br = 0, tr = 0, bl = 0;
    for (int i = 1; i < 4; ++i) {
        if (S(i) < S(tl)) tl = i;      
        if (S(i) > S(br)) br = i;       
        if (D(i) > D(tr)) tr = i;       
        if (D(i) < D(bl)) bl = i;       
    }
    return {tl, tr, br, bl}; // TL, TR, BR, BL
}


// quad: TL, TR, BR, BL (CCW)
inline float d2(const cv::Point2f& a, const cv::Point2f& b) {
    float dx = a.x - b.x, dy = a.y - b.y;
    return dx*dx + dy*dy;  
}


void printShortestLongest(const std::vector<cv::Point2f>& q) {
    float e0 = d2(q[0], q[1]);
    float e1 = d2(q[1], q[2]);
    float e2 = d2(q[2], q[3]);
    float e3 = d2(q[3], q[0]);

    float smin = std::min(std::min(e0, e1), std::min(e2, e3));
    float smax = std::max(std::max(e0, e1), std::max(e2, e3));

    float shortest = std::sqrt(smin);
    float longest  = std::sqrt(smax);

    std::cout << "Shortest: " << shortest << " px, Longest: " << longest << " px\n";
}

// Thread 1a: Capture camera frames using V4L2 non-blocking
void captureLoop(const string& device, FrameQueue& rawQ) {
  
    // Open device in non-blocking mode
    int fd = open(device.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        cerr << "Error opening " << device << ": " << strerror(errno) << endl;
        rawQ.push(Mat());
        return;
    }

    // Set format (YUYV, 640x480)
    v4l2_format fmt = {};
    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = 640;
    fmt.fmt.pix.height      = 480;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("VIDIOC_S_FMT");
        close(fd);
        rawQ.push(Mat());
        return;
    }

    // Request and mmap buffers
    v4l2_requestbuffers req = {};
    req.count  = 2; 
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("VIDIOC_REQBUFS");
        close(fd);
        rawQ.push(Mat());
        return;
    }

    struct Buffer { void* start; size_t length; };
    vector<Buffer> bufs(req.count);
    for (int i = 0; i < (int)req.count; ++i) {
        v4l2_buffer buf = {};
        buf.type   = req.type;
        buf.memory = req.memory;
        buf.index  = i;
        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("VIDIOC_QUERYBUF");
            close(fd);
            rawQ.push(Mat());
            return;
        }
        bufs[i].length = buf.length;
        bufs[i].start  = mmap(nullptr, buf.length,
                              PROT_READ | PROT_WRITE,
                              MAP_SHARED, fd, buf.m.offset);
        
        if (bufs[i].start == MAP_FAILED) {
            perror("mmap");
            close(fd);
            rawQ.push(Mat());
            return;
        }
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            perror("VIDIOC_QBUF");
            close(fd);
            rawQ.push(Mat());
            return;
        }
    }

    // Start streaming
    int type = req.type;
    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        perror("VIDIOC_STREAMON");
        close(fd);
        rawQ.push(Mat());
        return;
    }
    

    // Non-blocking capture loop
    v4l2_buffer buf = {};
    buf.type   = req.type;
    buf.memory = req.memory;
    while (true) {
        int ret = ioctl(fd, VIDIOC_DQBUF, &buf);
        if (ret < 0) {
            if (errno == EAGAIN) {
                this_thread::sleep_for(chrono::microseconds(100));
                continue;
            } else {
                perror("VIDIOC_DQBUF");
                break;
            }
        }

        // Wrap YUYV data into Mat and convert to BGR
        Mat yuyv(fmt.fmt.pix.height,
                 fmt.fmt.pix.width,
                 CV_8UC2,
                 bufs[buf.index].start);
        Mat bgr;
        cvtColor(yuyv, bgr, COLOR_YUV2BGR_YUYV);
        rawQ.push(bgr);

        // Re-queue buffer
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            perror("VIDIOC_QBUF");
            break;
        }
    }

    // Cleanup
    ioctl(fd, VIDIOC_STREAMOFF, &type);
    for (auto &b : bufs) munmap(b.start, b.length);
    close(fd);
    rawQ.push(Mat());
}


// Thread 1b: Input a Video File
void fileCaptureLoop(const string& filename, FrameQueue& rawQ) {
    VideoCapture cap(filename);
    if (!cap.isOpened()) {
        cerr << "Error opening file " << filename << endl;
        rawQ.push(Mat());
        return;
    }
    Mat frame;
    while (cap.read(frame)) {
        uint64_t start = now_ns();
        rawQ.push(frame);
        uint64_t dt_ns = elapsed_ns(start);
        
        double dt_ms = dt_ns/1e6;
        syslog(LOG_INFO, "Elapsed time for video capture loop: %.3f ms", dt_ms);
        if (waitKey(1) == 'q') break;
    }
    rawQ.push(Mat());
}


// Thread 2: Process frames
void processingLoop(FrameQueue& rawQ, FrameQueue& procQ) {

    static int frame_index = 0;
    vector<double> frame_times;
    Mat frame;


    while (rawQ.pop(frame)) {
    
        
        uint64_t start_ms = now_ns();

        auto start = chrono::high_resolution_clock::now();

        set<int> used_indices; // Keep track of all points used in any detected quad

        Mat hsv, gray, closed;

        // Run Image Transformation on GPU through CUDA
        g_cuda.run(frame, hsv, gray, closed);
        imshow("Transformed Image before Contour Detection", closed);

        // Find contours in the closed image
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(closed, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        // Detect Stickers and Classify Colors
        Mat output = frame.clone();
        vector<tuple<double,int,int>> seen_contours;
        vector<Center> centers;
        int idx = 1;

        static Point2f dbgP1, dbgP2, dbgP3;
        for (auto& cnt : contours) {
            double area = contourArea(cnt);
            if (area < 300 || area > 10000) continue;

            // shape approximation
            vector<Point> approx;
            approxPolyDP(cnt, approx, 0.05*arcLength(cnt,true), true);
            if (approx.size() > 8) continue;

            // mask & mean HSV
            Mat mask = Mat::zeros(frame.size(), CV_8U);
            drawContours(mask, vector<vector<Point>>{approx}, -1, 255, FILLED);
            
            Scalar m = mean(hsv, mask); // Returns H, S, V, A
            Vec3f hsvAvg((float)m[0], (float)m[1], (float)m[2]);

            // color classification
            string cname = classifyColor(hsvAvg);

            // centroid detection 
            Moments Mo = moments(approx);
            if (Mo.m00 == 0) continue;
            int cx = int(Mo.m10/Mo.m00), cy = int(Mo.m01/Mo.m00);
            if (isDuplicate(cx, cy, seen_contours)) continue;
            seen_contours.emplace_back(area, cx, cy);
            centers.push_back({idx, cx, cy, cname});

            // drawColor lookup
            Scalar drawColor(0,0,0);
            auto it = color_ranges.find(cname);
            if (it != color_ranges.end())
                drawColor = get<2>(it->second);

            idx++;
        }

        vector<Face> faces; // Initialize before the loop that finds each face

        // Check for every Triplet to build and validate Quad 
        int n = (int)centers.size();
        for (int p = 1; p <= n; ++p) {
            if (used_indices.count(p)) continue;
            Center* p1 = findByIndex(centers, p);
            if (!p1) continue;
            for (int q = p+1; q <= n; ++q) {
                if (used_indices.count(q)) continue;    
                Center* p2 = findByIndex(centers, q);
                if (!p2) continue;
                for (int r = q+1; r <= n; ++r) {
                    if (used_indices.count(r)) continue;
                    Center* p3 = findByIndex(centers, r);
                    if (!p3) continue;

                    // Collinearity test via triangle area
                    double A = triangleArea(
                        p1->cx, p1->cy,
                        p2->cx, p2->cy,
                        p3->cx, p3->cy
                    );
                    if (A >= AREA_TOLERANCE) 
                        continue;

                    // Identify the longest of the three sides 
                    struct Trip { double d; Center* a; Center* b; Center* c; };
                    vector<Trip> trips = {
                        { dist(*p1, *p2), p1, p2, p3 },
                        { dist(*p2, *p3), p2, p3, p1 },
                        { dist(*p1, *p3), p1, p3, p2 },
                    };
                    auto best = *max_element(trips.begin(), trips.end(),
                        [](auto &L, auto &R){ return L.d < R.d; }
                    );
                    Center* start  = best.a;
                    Center* end    = best.b;
                    Center* middle = best.c;

                    // Try to build a quad from that base edge
                    auto firstLine = make_tuple(start, end, middle);
                    auto quadLines = buildQuadFromStartLine(
                        centers,
                        firstLine,
                        AREA_TOLERANCE
                    );
                    if (quadLines.empty())
                        continue;

                    // Extract the four corner points
                    vector<Center*> quadPts;
                    quadPts.reserve(4);
                    for (auto &ln : quadLines)
                        quadPts.push_back(get<0>(ln));

                    // Validate and draw
                    if (quadPts.size() == 4 && isParallelogram(quadPts) && isMiddleBalanced(quadLines)) {
                    
                        drawQuadOnFrame(
                            output,
                            quadPts[0], quadPts[1],
                            quadPts[2], quadPts[3],
                            Scalar(255,0,255), 2
                        );

                        set<int> quad_indices;
                        for (const auto& ln : quadLines) {
                            quad_indices.insert(get<0>(ln)->index);  
                            quad_indices.insert(get<1>(ln)->index);  
                            quad_indices.insert(get<2>(ln)->index);  
                        }

                        // Add them to the global used_indices set
                        used_indices.insert(quad_indices.begin(), quad_indices.end());

                        // Draw each point in the quad with its detected color
                        for (int idx_used : quad_indices) {
                            const Center* pt = findByIndex(centers, idx_used);
                            if (!pt) continue;

                            Scalar bgr_color(0,0,0); 
                            auto it = color_ranges.find(pt->hsv_color_name);
                            if (it != color_ranges.end())
                                bgr_color = get<2>(it->second);      

                            circle(output, Point(pt->cx, pt->cy), 8, bgr_color, FILLED);
                        }
                        
                        // Get polygon vertices as (x, y) tuples
                        vector<Point> poly_pts;
                        poly_pts.reserve(4);
                        for (int i = 0; i < 4 && i < static_cast<int>(quadPts.size()); ++i)
                            poly_pts.emplace_back(quadPts[i]->cx, quadPts[i]->cy);

                        // Store the centers that belong to the face
                        vector<Center*> inside_centers;
                        vector<int> inside_indices;

                        // Find centers inside the parallelogram
                        for (auto &c : centers) {
                            double res = pointPolygonTest(poly_pts, Point2f(static_cast<float>(c.cx),
                                                                                    static_cast<float>(c.cy)), false);
                            if (res >= 0) { 
                                inside_indices.push_back(c.index);
                                inside_centers.push_back(&c);
                            }
                        }

                        if (!inside_centers.empty()) {
                            long sumx = 0, sumy = 0;
                            for (const Center* c : inside_centers) {
                                sumx += c->cx;
                                sumy += c->cy;
                            }
                            int mean_x = static_cast<int>(sumx / static_cast<long>(inside_centers.size()));
                            int mean_y = static_cast<int>(sumy / static_cast<long>(inside_centers.size()));
                            faces.push_back(Face{Point(mean_x, mean_y), inside_indices });
                        }
                        
                        // Get the centers for all stickers belonging to the face from quad_indices
                        vector<Center*> face_stickers;
                        face_stickers.reserve(quad_indices.size());
                        for (int idx_used : quad_indices) {
                            Center* p = findByIndex(centers, idx_used);
                            if (p) face_stickers.push_back(p);
                        }

                        // Order corners and assign positions by parallelogram
                        vector<Center*> ordered_corners = orderParallelogramCorners(quadPts); // TL, TR, BL, BR
                        auto positions = assignPositionsByParallelogram(face_stickers, ordered_corners);
                        
                        for (const auto& kv : positions) {
                            int idx = kv.first;
                            int row = kv.second.first;
                            int col = kv.second.second;

                            // find the sticker center with this index inside the face
                            Center* center = nullptr;
                            for (Center* c : face_stickers) {
                                if (c->index == idx) { center = c; break; }
                            }
                            if (!center) continue;

                            Scalar bgr_color(255, 255, 255);
                            auto it = color_ranges.find(center->hsv_color_name);
                            if (it != color_ranges.end())
                                bgr_color = get<2>(it->second);

                            circle(output, Point(center->cx, center->cy), 6, bgr_color, FILLED);
                            
                            double brightness = (bgr_color[0] + bgr_color[1] + bgr_color[2]) / 3.0;
                            Scalar text_color = (brightness > 127.0) ? Scalar(0,0,0) : Scalar(255,255,255);

                            putText(output, to_string(row) + "," + to_string(col),
                                Point(center->cx - 10, center->cy - 10),
                                FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1);

                        }

                        SideInfo sideInfo = getParallelogramSideInfo(ordered_corners);
                        
                        double shortest_side = sideInfo.len[sideInfo.shortest];
                        
                        if (shortest_side < MIN_SIDE_OK) {
                            cout << "[DISTANCE WARNING] Cube too far — move closer\n";
                            cv::putText(output, "MOVE CLOSER",
                                        Point(30,30), FONT_HERSHEY_SIMPLEX, 0.8,
                                        Scalar(0,0,255), 2);
                        }
                        else if (shortest_side > MAX_SIDE_OK) {
                            cout << "[DISTANCE WARNING] Cube too near — move away\n";
                            cv::putText(output, "MOVE AWAY",
                                        Point(30,30), FONT_HERSHEY_SIMPLEX, 0.8,
                                        Scalar(0,0,255), 2);
                        }
                        else {
                            cout << "[DISTANCE OK] Perfect distance\n";
                            cv::putText(output, "GOOD DISTANCE",
                                        Point(30,30), FONT_HERSHEY_SIMPLEX, 0.8,
                                        Scalar(0,255,0), 2);
                        }
                        
                        // After all three faces have been collected:
                        if (faces.size() == 3) {
                            auto it_top = min_element(faces.begin(), faces.end(),
                                [](const Face& a, const Face& b){ return a.center.y < b.center.y; });
                            const Face* top_face = &(*it_top);

                            vector<const Face*> remaining;
                            remaining.reserve(2);
                            for (const auto& f : faces)
                                if (&f != top_face) remaining.push_back(&f);

                            auto it_right = max_element(
                                remaining.begin(), remaining.end(),
                                [](const Face* a, const Face* b){ return a->center.x < b->center.x; });
                            const Face* right_face = *it_right;

                            const Face* front_face = (remaining[0] == right_face) ? remaining[1] : remaining[0];

                            for (int idx : front_face->indices) {
                                Center* sticker = findByIndex(centers, idx);
                                if (!sticker) continue;

                                int cx = sticker->cx, cy = sticker->cy;
                                
                                // Classified color name in hsv_color_name
                                cout << "Index " << idx
                                        << ": COLOR=" << sticker->hsv_color_name
                                        << " -> (" << cx << ", " << cy << ")\n";
                            }

                            // Print index lists
                            auto print_indices = [](const char* label, const std::vector<int>& v){
                                std::cout << label << ": ";
                                for (int i : v) std::cout << i << ' ';
                                std::cout << '\n';
                            };

                            print_indices("TOP",   top_face->indices);
                            print_indices("FRONT", front_face->indices);
                            print_indices("RIGHT", right_face->indices);

                            
                            cv::putText(output, "TOP",   top_face->center,
                                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
                            cv::putText(output, "FRONT", front_face->center,
                                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
                            cv::putText(output, "RIGHT", right_face->center,
                                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
                        }

                        else{
                            cout << "Only " << faces.size() << " faces detected this frame\n";
                        }
                        

                    }    
                                            
                }
            }
        }

        // Push Processed Frame and Record Time
        procQ.push(output);
        auto end = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(end - start).count();
        frame_times.push_back(elapsed);
        ++frame_index;

        uint64_t dt_ns = elapsed_ns(start_ms);
        double dt_ms = dt_ns/1e6;
        syslog(LOG_INFO, "Elapsed time for processing loop: %.3f ms", dt_ms);
    }
  
    // Signal end of processing
    procQ.push(Mat());

    // Print Frame Statistics
    if (!frame_times.empty()) {
    
        const size_t N = frame_times.size();

        const double sum = accumulate(frame_times.begin(), frame_times.end(), 0.0);
        const double avg_time = sum / static_cast<double>(N);
        const double avg_fps  = (avg_time > 0.0) ? 1.0 / avg_time : 0.0;

        double worst_time = *max_element(frame_times.begin(), frame_times.end());
        double min_fps    = (worst_time > 0.0) ? 1.0 / worst_time : 0.0;

        cout << "\nFrames processed: " << frame_times.size();
        cout << "\nAverage frame time: " << avg_time << " s";
        cout << "\nAverage FPS: " << avg_fps;
        cout << "\nWorst-case frame time: " << worst_time << " s";
        cout << "\nMin FPS (from worst frame): " << min_fps << "\n";
        
    } else {
        cout << "No frames processed." << endl;
    }
}

// Thread 3: Display results
void displayLoop(FrameQueue& procQ) {
    Mat out;
    while (procQ.pop(out)) {
      
        uint64_t start_ms = now_ns();
    
    
        imshow("Rubik's Cube Tracking", out);
        
        uint64_t dt_ns = elapsed_ns(start_ms);
        
        double dt_ms = dt_ns/1e6;
        syslog(LOG_INFO, "Elapsed time for display loop: %.3f ms", dt_ms);
        
        if (waitKey(1) == 'q') break;
    }
}


int main(int argc, char** argv) {

    openlog("CubeTracker", LOG_PID | LOG_CONS, LOG_USER);
    if (argc != 3 || (string(argv[1])!="-f" && string(argv[1])!="-d")) {
        cerr << "Usage: " << argv[0] << " -f <video_file> | -d <device>\n";
        return -1;
    }
    
    string mode = argv[1], source = argv[2];
    FrameQueue rawQ, procQ;
    thread tCap;
    
    if (mode == "-f") 
        tCap = thread(fileCaptureLoop, source, ref(rawQ));
    else              
        tCap = thread(captureLoop, source, ref(rawQ));

    thread tProc(processingLoop, ref(rawQ), ref(procQ));
    thread tDisp(displayLoop, ref(procQ));
    tCap.join(); tProc.join(); tDisp.join();
    
    closelog();

    return 0;
}
