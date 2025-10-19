
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <set>
#include <map>
#include <string>

using namespace cv;
using namespace std;

static constexpr double AREA_TOLERANCE = 400.0;

// struct to hold each sticker center
struct Center {
    int index;
    int cx, cy;
    string hsv_color_name;
};

struct Face {
    Point center;
    vector<int> indices;
};

// Color‐range map (HSV lower, HSV upper, BGR draw color)
extern const map<string, tuple<Scalar,Scalar,Scalar>> color_ranges;

// Geometry & color‐classification helpers
Vec3f    getAverageColor(const Scalar& lo, const Scalar& hi);
string   classifyColor(const Vec3f& hsvAvg);
bool     isDuplicate(int cx, int cy, const vector<tuple<double,int,int>>& seen, int thresh=10);

Center*  findByIndex(vector<Center>& centers, int idx);
double   triangleArea(int x1,int y1,int x2,int y2,int x3,int y3);
double   dist(const Center& p1, const Center& p2);
double   angleBetween(const Point2d& v1,const Point2d& v2);

vector<tuple<Center*,Center*,Center*>>
         findAllNextCollinearSides(
           vector<Center>& centers,
           Center* fixedPoint,
           double areaTol=AREA_TOLERANCE,
           const set<int>& usedPoints = {},
           Center* allowPoint = nullptr);

vector<tuple<Center*,Center*,Center*>>
         buildQuadFromStartLine(
           vector<Center>& centers,
           const tuple<Center*,Center*,Center*>& firstLine,
           double areaTol = AREA_TOLERANCE,
           double sideRatioThresh = 1.5,
           const pair<double,double>& angleThresh = {30.0,150.0},
           set<int> usedPoints = {},
           vector<tuple<Center*,Center*,Center*>> lines = {},
           int depth = 0);

bool     isParallelogram(
           const vector<Center*>& pts,
           double lengthRatioTol = 0.25,
           double angleTolDeg    = 15.0);

bool     isMiddleBalanced(
           const vector<tuple<Center*,Center*,Center*>>& lines,
           double ratioThresh = 1.3);

void     drawQuadOnFrame(
           Mat& frame,
           Center* p1, Center* p2, Center* p3, Center* p4,
           const Scalar& color = Scalar(255,0,255),
           int thickness = 2);

vector<Center*> orderParallelogramCorners(
           const std::vector<Center*>& pts);

unordered_map<int, pair<int,int>>
assignPositionsByParallelogram(const vector<Center*>& stickers,
                               const vector<Center*>& corners);