#include "helper_funcs.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

// Color Range Map
const map<string, tuple<Scalar,Scalar,Scalar>> color_ranges = {
    {"white",  { Scalar(  0,   0, 200), Scalar(180,  50, 255), Scalar(255, 255, 255) }},
    {"yellow", { Scalar( 20, 100, 100), Scalar( 35, 255, 255), Scalar(  0, 255, 255) }},
    {"orange", { Scalar( 10, 100, 100), Scalar( 20, 255, 255), Scalar(  0, 165, 255) }},
    {"blue",   { Scalar( 90, 100, 100), Scalar(130, 255, 255), Scalar(255,   0,   0) }},
    {"green",  { Scalar( 40,  70, 100), Scalar( 85, 255, 255), Scalar(  0, 255,   0) }},
    {"red1",   { Scalar(  0, 100, 100), Scalar( 10, 255, 255), Scalar(  0,   0, 255) }},  // red lower hue
    {"red2",   { Scalar(160, 100, 100), Scalar(180, 255, 255), Scalar(  0,   0, 255) }}   // red upper hue
};

// Compute the midpoint of two HSV Scalar bounds
Vec3f getAverageColor(const Scalar& lower, const Scalar& upper) {
    return Vec3f(
        (lower[0] + upper[0]) * 0.5f,
        (lower[1] + upper[1]) * 0.5f,
        (lower[2] + upper[2]) * 0.5f
    );
}

// Classify an HSV mean into the closest color in color_ranges
string classifyColor(const Vec3f& hsvAvg) {
    double minDist = numeric_limits<double>::infinity();
    string best = "unknown";

    for (const auto& kv : color_ranges) {
        const string& name    = kv.first;
        const Scalar& lower   = get<0>(kv.second);
        const Scalar& upper   = get<1>(kv.second);

        Vec3f reference = getAverageColor(lower, upper);
        double d = norm(hsvAvg - reference);

        if (d < minDist) {
            minDist = d;
            best    = name;
        }
    }

    return best;
}

// Check for Duplicate Contours
bool isDuplicate(int cx, int cy, const vector<tuple<double,int,int>>& seen, int thresh) {
    for (auto& t : seen) {
        int sx = get<1>(t), sy = get<2>(t);
        if (abs(cx - sx) < thresh && abs(cy - sy) < thresh)
            return true;
    }
    return false;
}

// Find by Index Function
Center* findByIndex(vector<Center>& centers, int idx) {
    for (auto& c : centers) {
        if (c.index == idx) return &c;
    }
    return nullptr;
}

// Triangle Area Function
double triangleArea(int x1, int y1, int x2, int y2, int x3, int y3) {
    return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2));
}

// Distance between Centers Function
double dist(const Center& p1, const Center& p2) {
    return hypot(p2.cx - p1.cx, p2.cy - p1.cy);
}

// Angle Between Vectors
double angleBetween(const Point2d& v1, const Point2d& v2) {
    double dot = v1.dot(v2);
    double mag1 = norm(v1);
    double mag2 = norm(v2);
    if (mag1 == 0 || mag2 == 0) return 0;
    double cosTheta = clamp(dot / (mag1 * mag2), -1.0, 1.0);
    return acos(cosTheta) * 180.0 / CV_PI;
}

// Find next Collinear Side Function
// Returns all (start, end, middle) triples of points nearly collinear with fixedPoint
vector<tuple<Center*, Center*, Center*>> findAllNextCollinearSides(
    vector<Center>& centers,
    Center* fixedPoint,
    double areaTol,
    const set<int>& usedPoints,
    Center* allowPoint)
{
    // build candidate list
    vector<Center*> candidates;
    for (auto& c : centers) {
        if (&c == fixedPoint) continue;
        if (usedPoints.count(c.index)) {
            if (allowPoint && allowPoint->index == c.index)
                candidates.push_back(&c);
        } else {
            candidates.push_back(&c);
        }
    }

    vector<tuple<Center*, Center*, Center*>> results;
    // every pair (p, q) among candidates
    for (size_t i = 0; i < candidates.size(); ++i) {
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            Center* p = candidates[i];
            Center* q = candidates[j];

            // collinearity via small triangle area
            double A = triangleArea(
                fixedPoint->cx, fixedPoint->cy,
                p->cx, p->cy,
                q->cx, q->cy
            );
            if (A >= areaTol) continue;

            // compute the three segment lengths
            double d1 = dist(*fixedPoint, *p);
            double d2 = dist(*p, *q);
            double d3 = dist(*fixedPoint, *q);

            struct Trip { double d; Center* a; Center* b; Center* c; };
            vector<Trip> trips = {
                {d1, fixedPoint, p, q},
                {d2, p, q, fixedPoint},
                {d3, fixedPoint, q, p}
            };

            // pick the longest segment (start, end, middle)
            auto best = max_element(
                trips.begin(), trips.end(),
                [](auto &L, auto &R){ return L.d < R.d; }
            );
            Center* start  = best->a;
            Center* end    = best->b;
            Center* middle = best->c;

            // ensure fixedPoint is one endpoint
            if (fixedPoint != start && fixedPoint != end)
                continue;
            if (start != fixedPoint)
                swap(start, end);

            results.emplace_back(start, end, middle);
        }
    }

    return results;
}


// Build Quad from Start Line Function
// Returns an empty vector on failure, or the list of 4 collinear‐side triples on success.
vector<tuple<Center*,Center*,Center*>> buildQuadFromStartLine(
    vector<Center>& centers,
    const tuple<Center*,Center*,Center*>& firstLine,
    double areaTol,
    double sideRatioThresh,
    const pair<double,double>& angleThresh,
    set<int> usedPoints,
    vector<tuple<Center*,Center*,Center*>> lines,
    int depth)
{
    if (depth == 0) {
        usedPoints = {
            get<0>(firstLine)->index,
            get<1>(firstLine)->index,
            get<2>(firstLine)->index
        };
        lines = { firstLine };
    }

    // Early angle check when depth>=2
    if (depth >= 2) {
        int sz = (int)lines.size();
        auto &tpl_prev = lines[sz-2];
        Center *p_prev = get<0>(tpl_prev), *p_curr = get<1>(tpl_prev);
        auto &tpl_curr = lines[sz-1];
        Center *p_next = get<1>(tpl_curr);

        Point2d v1(p_prev->cx - p_curr->cx, p_prev->cy - p_curr->cy);
        Point2d v2(p_next->cx - p_curr->cx, p_next->cy - p_curr->cy);
        double ang = angleBetween(v1, v2);
        if (ang < angleThresh.first || ang > angleThresh.second)
            return {};
    }

    // Completion check at depth==3
    if (depth == 3) {
        auto &last_tpl  = lines.back();
        Center *last_end   = get<1>(last_tpl);
        auto &first_tpl = lines.front();
        Center *first_start = get<0>(first_tpl);
        if (last_end->index != first_start->index) return {};

        // Side‐length ratio
        vector<double> sides;
        sides.reserve(4);
        for (auto &ln : lines) {
            Center *a = get<0>(ln), *b = get<1>(ln);
            sides.push_back(dist(*a, *b));
        }
        auto [minIt, maxIt] = minmax_element(sides.begin(), sides.end());
        if (*maxIt > sideRatioThresh * *minIt) return {};

        // Build unique corner list
        vector<Center*> quad;
        quad.reserve(4);
        for (auto &ln : lines) {
            Center *a = get<0>(ln);
            if (quad.empty() || quad.back()->index != a->index)
                quad.push_back(a);
        }
        if (get<1>(lines.back())->index != quad.front()->index)
            quad.push_back(get<1>(lines.back()));
        if ((int)quad.size() != 4) return {};

        // Final corner‐angle test
        for (int i = 0; i < 4; ++i) {
            Center *prev = quad[(i+3)%4], *curr = quad[i], *next = quad[(i+1)%4];
            Point2d va(prev->cx - curr->cx, prev->cy - curr->cy);
            Point2d vb(next->cx - curr->cx, next->cy - curr->cy);
            double a = angleBetween(va, vb);
            if (a < angleThresh.first || a > angleThresh.second)
                return {};
        }
        return lines;
    }

    // Extend the quad by finding collinear sides off the last endpoint
    Center* fixedPoint = get<1>(lines.back());
    auto nextLines = findAllNextCollinearSides(
        centers,
        fixedPoint,
        areaTol,
        usedPoints,
        depth == 2 ? get<0>(firstLine) : nullptr
    );

    for (auto &nl : nextLines) {
        Center *endPt = get<1>(nl);
        if (usedPoints.count(endPt->index) && endPt != get<0>(firstLine))
            continue;

        auto newUsed  = usedPoints;
        newUsed.insert(endPt->index);

        auto newLines = lines;
        newLines.push_back(nl); 

        auto result = buildQuadFromStartLine(
            centers,
            firstLine,
            areaTol,
            sideRatioThresh,
            angleThresh,
            newUsed,
            newLines,
            depth + 1
        );
        if (!result.empty())
            return result;
    }

    return {};
}


// Checks if a parallelogram is formed with the given points
bool isParallelogram(
    const vector<Center*>& pts,
    double lengthRatioTol,
    double angleTolDeg)
{
    if (pts.size() != 4)
        throw invalid_argument("Exactly 4 points required");

    Center *A = pts[0], *B = pts[1], *C = pts[2], *D = pts[3];

    // build the four side vectors
    Point2d AB(B->cx - A->cx, B->cy - A->cy);
    Point2d BC(C->cx - B->cx, C->cy - B->cy);
    Point2d CD(D->cx - C->cx, D->cy - C->cy);
    Point2d DA(A->cx - D->cx, A->cy - D->cy);

    // lengths
    double lenAB = norm(AB), lenBC = norm(BC);
    double lenCD = norm(CD), lenDA = norm(DA);

    // side‐length ratio checks
    double r1 = lenCD / (lenAB + 1e-10);
    if (r1 < 1 - lengthRatioTol || r1 > 1 + lengthRatioTol)
        return false;

    double r2 = lenDA / (lenBC + 1e-10);
    if (r2 < 1 - lengthRatioTol || r2 > 1 + lengthRatioTol)
        return false;

    // angle between opposite sides
    double angAB_CD = angleBetween(AB, CD);
    if (!(angAB_CD < angleTolDeg || abs(angAB_CD - 180.0) < angleTolDeg))
        return false;

    double angBC_DA = angleBetween(BC, DA);
    if (!(angBC_DA < angleTolDeg || abs(angBC_DA - 180.0) < angleTolDeg))
        return false;

    return true;
}


bool isMiddleBalanced(
    const vector<tuple<Center*,Center*,Center*>>& lines,
    double ratioThresh)
{
    for (size_t i = 0; i < lines.size(); ++i) {
        Center* start  = get<0>(lines[i]);
        Center* end    = get<1>(lines[i]);
        Center* middle = get<2>(lines[i]);

        double d1 = dist(*start, *middle);
        double d2 = dist(*end,   *middle);

        double shorter = min(d1, d2);
        double longer  = max(d1, d2);

        if (shorter < 1e-3) {
            cout << "Line " << (i+1) << ": Too short to compare reliably." << endl;
            return false;
        }

        double ratio = longer / shorter;
        if (ratio > ratioThresh) {
            cout << "Line " << (i+1)
                 << " failed middle-balance check: ratio = "
                 << ratio << " (limit " << ratioThresh << ")" << endl;
            return false;
        }
    }
    return true;
}

// Draw Quad on Frame Function, connecting 4 centers on frame
void drawQuadOnFrame(
    Mat& frame,
    Center* p1, Center* p2, Center* p3, Center* p4,
    const Scalar& color,
    int thickness)
{
    // Gather the four corner points
    vector<Point> pts = {
        Point(p1->cx, p1->cy),
        Point(p2->cx, p2->cy),
        Point(p3->cx, p3->cy),
        Point(p4->cx, p4->cy)
    };

    // Compute centroid as float sum
    Point2f center(0.f, 0.f);
    for (const auto& pt : pts) {
        center.x += static_cast<float>(pt.x);
        center.y += static_cast<float>(pt.y);
    }
    center.x /= static_cast<float>(pts.size());
    center.y /= static_cast<float>(pts.size());

    // Sort points by angle around centroid (clockwise)
    sort(pts.begin(), pts.end(),
        [&center](const Point& a, const Point& b) {
            double angA = atan2(a.y - center.y, a.x - center.x);
            double angB = atan2(b.y - center.y, b.x - center.x);
            return angA < angB;
        }
    );

    // Draw the closed quadrilateral
    vector<vector<Point>> contour = { pts };
    polylines(frame, contour, true, color, thickness);
}


vector<Center*> orderParallelogramCorners(const vector<Center*>& pts) {
    if (pts.size() != 4) throw std::invalid_argument("Need exactly 4 points");

    // Build array of Point2f
    vector<Point2f> P;
    P.reserve(4);
    for (auto* c : pts) P.emplace_back(c->cx, c->cy);

    // Convex hull indices
    vector<int> hullIdx;
    convexHull(P, hullIdx, false, false);
    if ((int)hullIdx.size() != 4) throw runtime_error("Need 4 hull points");

    // Centroid of hull
    Point2f c(0.f, 0.f);
    for (int i : hullIdx) c += P[i];
    c.x /= 4.0f; c.y /= 4.0f;

    // Sort hull by angle around centroid
    sort(hullIdx.begin(), hullIdx.end(), [&](int i, int j){
        const auto& a = P[i]; const auto& b = P[j];
        double ai = std::atan2(a.y - c.y, a.x - c.x);
        double aj = std::atan2(b.y - c.y, b.x - c.x);
        return ai < aj; 
    });

    // Rotate list to start at Top Left
    auto tl_it =min_element(hullIdx.begin(), hullIdx.end(), [&](int i, int j){
        if (P[i].y != P[j].y) return P[i].y < P[j].y;      
        return P[i].x < P[j].x;                           
    });
    // rotate so TL is first
    rotate(hullIdx.begin(), tl_it, hullIdx.end()); 

    // Map back to pointers and return
    vector<Center*> ordered = { pts[hullIdx[0]], pts[hullIdx[1]], pts[hullIdx[2]], pts[hullIdx[3]] };
    return ordered; 
}



unordered_map<int, pair<int,int>>
assignPositionsByParallelogram(const vector<Center*>& stickers,
                               const vector<Center*>& corners)
{
    if (corners.size() != 4)
        throw invalid_argument("corners must be [p00, p02, p20, p22]");

    const Center* p00 = corners[0]; 
    const Center* p02 = corners[1]; 
    const Center* p20 = corners[2]; 

    // matrix = column_stack((vec_top, vec_left))
    const double a = double(p02->cx - p00->cx); 
    const double c = double(p02->cy - p00->cy); 
    const double b = double(p20->cx - p00->cx); 
    const double d = double(p20->cy - p00->cy); 
    const double det = a*d - b*c;

    
    if (abs(det) < 1e-6) return {};

    unordered_map<int, pair<int,int>> positions;
    positions.reserve(stickers.size());

    for (const Center* s : stickers) {
        const double rx = double(s->cx - p00->cx);
        const double ry = double(s->cy - p00->cy);

        double u = ( d*rx - b*ry) / det;
        double v = (-c*rx + a*ry) / det;

        u = clamp(u, 0.0, 1.0);
        v = clamp(v, 0.0, 1.0);

        // Map to 3x3 grid
        int col = int(lround(u * 2.0));
        int row = int(lround(v * 2.0));
        col = clamp(col, 0, 2);
        row = clamp(row, 0, 2);

        positions[s->index] = {row, col};
    }

    return positions;
}

