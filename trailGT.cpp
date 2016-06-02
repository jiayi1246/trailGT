#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <set>

#include "dirent.h"

#include <unistd.h>
#include <sys/times.h>
#include <sys/time.h>

using namespace std;
using namespace cv;

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

void add_all_images();
void add_images(string, vector <string> &);
void draw_overlay();
void onMouse(int, int, int, int, void *);
void saveBad();
void loadBad();

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

string imdirname = "/warthog_logs/Aug_05_2011_Fri_11_50_24_AM/omni_images/";
string fname;

vector <string> dir_image_filename;  // full-path image filenames
//vector <GT *> image_GT;   
 
Mat current_im, draw_im;

vector <int> Random_idx;         // R[i] holds random index
vector <int> Nonrandom_idx;      // R[N[i]] = i
set <int> Bad_idx_set;

bool do_random = true;
bool do_overlay = true;
bool do_bad = false;
int bad_start, bad_end;          // indices of bad range

bool callbacks_set = false;
bool dragging = false;
int dragging_x, dragging_y;

double fontScale = 0.35;

vector <int> trailEdgeRow;

int current_index = 0;
string current_imname;

//----------------------------------------------------------------------------

// if being run for first time: 
//  -- go through directories and gather image names
//  -- write some kinda database file with names, "empty" GT info (serialize map with OpenCV's XML output capabilities?) 
// if database file exists, read it and print info (how many images, how many have GT)

// pick a random image (subject to constraints, like not too close to existing choice) and show it
// UI for picking point(s) [such as constraint on Y val], saving to GT

// mode for showing existing GT points
// save key to re-write entire database

//----------------------------------------------------------------------------

// get all jpg/png image names in directory dirname, append them to filenames string vector

void add_images(string dirname, vector <string> & imnames)
{
  string imname;
  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir (dirname.c_str())) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      if (strstr(ent->d_name, "jpg") != NULL || strstr(ent->d_name, "png") != NULL) {
	
	imname = dirname + string(ent->d_name);
	imnames.push_back(imname);
	//	printf("read %s\n", imname.c_str());


      }
    }
    closedir (dir);
  } 
}

//----------------------------------------------------------------------------

void add_all_images()
{
  add_images(string("/warthog_logs/Aug_05_2011_Fri_11_50_24_AM/omni_images/"), dir_image_filename);
  add_images(string("/warthog_logs/Aug_05_2011_Fri_11_54_36_AM/omni_images/"), dir_image_filename);
  add_images(string("/warthog_logs/Aug_05_2011_Fri_12_25_11_PM/omni_images/"), dir_image_filename);
  add_images(string("/warthog_logs/Aug_05_2011_Fri_12_29_30_PM/omni_images/"), dir_image_filename);

  //  add_images(string("./test_omni_images/"), dir_image_filename);

  sort(dir_image_filename.begin(), dir_image_filename.end());
}

//----------------------------------------------------------------------------

// return closest guide row

int snap_y(int y)
{
  int diff, min_diff, min_y;
  min_diff = 10000;

  for (int i = 0; i < trailEdgeRow.size(); i++) {
    diff = abs(y - trailEdgeRow[i]);
    if (diff < min_diff) {
      min_diff = diff;
      min_y = trailEdgeRow[i];
    }
  }

  return min_y;
}

//----------------------------------------------------------------------------

void onMouse(int event, int x, int y, int flags, void *userdata)
{
  int vx, vy;
  int g, b;

  if  ( event == EVENT_MOUSEMOVE ) {

    vx = x;
    if (dragging) {
      g = 200;
      b = 0;
      vy = dragging_y;
    }
    else {
      g = 255; 
      b = 255;
      vy = snap_y(y);
    }
  }

  else if  ( event == EVENT_LBUTTONDOWN ) {

    dragging = true;

    g = 255;
    b = 0;
    vx = x;
    vy = snap_y(y);
    dragging_x = vx;
    dragging_y = vy;

  }

  else if  ( event == EVENT_LBUTTONUP ) {

    dragging = false;

    g = 255;
    b = 0;
    vx = x;
    vy = dragging_y;

  }

  draw_im = current_im.clone();
  draw_overlay();
  if (dragging)
    line(draw_im, Point(dragging_x, vy), Point(vx, vy), Scalar(0, 255, 0), 2);
  circle(draw_im, Point (vx, vy), 8, Scalar(0, g, b), 1, 8, 0);
  imshow("trailGT", draw_im);  

  /*
  if  ( event == EVENT_LBUTTONDOWN ) {
    cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    drawing=true;
    circle(gimg,Point (x,y) ,6, Scalar(0,0,255),3,8,0);
    imshow("trailGT",gimg);  
  }
  else if  ( event == EVENT_RBUTTONUP ) {
    cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    printf("Trying to write to %s\n", fname.c_str()); fflush(stdout);
    //imwrite("/Users/jiayi/Desktop/trailGT1/warthog_logs/Aug_05_2011_Fri_11_50_24_AM/omni_images_saved/"+fname, gimg);
  }
  */
}

//----------------------------------------------------------------------------

void onKeyPress(char c)
{
  // goto next image in sequence
  
  if (c == 'n' || c == ']' || c == 'x') {
    
    if (!do_random) {

      current_index++;  
      
      // wrap?
      
      if (current_index >= dir_image_filename.size())
	current_index = 0;
    }
    else {
      current_index = Nonrandom_idx[current_index] + 1;  

      if (current_index >= dir_image_filename.size())
	current_index = 0;

      current_index = Random_idx[current_index];
    }

  }
  
  // goto previous image in sequence
  
  else if (c == 'p' || c == '[' || c == 'z') {
    
    if (!do_random) {

      current_index--;
      
      // wrap?
      
      if (current_index < 0)
	current_index = dir_image_filename.size() - 1;
    }
    else {
      current_index = Nonrandom_idx[current_index] - 1;  

      if (current_index < 0)
	current_index = dir_image_filename.size() - 1;

      current_index = Random_idx[current_index];
    }   
  }
  
  // toggle overlay
  
  else if (c == 'o') 
    do_overlay = !do_overlay;
  
  // toggle randomized index mode
  
  else if (c == 'r') 
    do_random = !do_random;

  // this is a bad image (no trail or trail geometry is wacky)
  
  else if (c == 'b') {
    if (do_random)
      return;
    do_bad = !do_bad;
    if (do_bad)
      bad_start = current_index;
    else {
      bad_end = current_index;
      for (int i = bad_start; i <= bad_end; i++)
	Bad_idx_set.insert(i);
    }
  }

  // allow bad images to become good again
  
  else if (c == 'g') {
    set<int>::iterator iter;
    iter = Bad_idx_set.find(current_index);
    if (iter != Bad_idx_set.end())
      Bad_idx_set.erase(iter);
  }

  // save

  else if (c == 's') {
    saveBad();
  }
}

//----------------------------------------------------------------------------

// set all mat values at given channel to given value

// from: http://stackoverflow.com/questions/23510571/how-to-set-given-channel-of-a-cvmat-to-a-given-value-efficiently-without-chang

void setChannel(Mat &mat, unsigned int channel, unsigned char value)
{
  // make sure have enough channels
  if (mat.channels() < channel + 1)
    return;
  
  const int cols = mat.cols;
  const int step = mat.channels();
  const int rows = mat.rows;
  for (int y = 0; y < rows; y++) {
    // get pointer to the first byte to be changed in this row
    unsigned char *p_row = mat.ptr(y) + channel; 
    unsigned char *row_end = p_row + cols*step;
    for (; p_row != row_end; p_row += step)
      *p_row = value;
  }
}

//----------------------------------------------------------------------------

void draw_overlay()
{
  if (do_overlay) {
    
    // which image is this?

    stringstream ss;
    ss << current_index << ": " << current_imname;
    string str = ss.str();

    putText(draw_im, str, Point(5, 10), FONT_HERSHEY_SIMPLEX, fontScale, Scalar::all(255), 1, 8);
    
    // are we in "random next image" mode?

    if (do_random) 
      putText(draw_im, "R", Point(5, 25), FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 255), 1, 8);

    // are we in "bad image" mode?

    if (do_bad) 
      putText(draw_im, "B", Point(15, 25), FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 255), 1, 8);

    // "bad" image?

    if (Bad_idx_set.find(current_index) != Bad_idx_set.end()) {
      setChannel(draw_im, 2, 255);
      return;
    }

    // horizontal lines for trail edge rows

    for (int i = 0; i < trailEdgeRow.size(); i++) 
      line(draw_im, Point(0, trailEdgeRow[i]), Point(current_im.cols - 1, trailEdgeRow[i]), Scalar(0, 128, 128), 1);
  }
}

//----------------------------------------------------------------------------

void saveBad()
{
  FILE *fp = fopen("bad.txt", "w");
  for (int i = 0; i < dir_image_filename.size(); i++) {
    bool bad = (Bad_idx_set.find(i) != Bad_idx_set.end());
    fprintf(fp, "%i: %i\n", i, bad);
    fflush(fp);
  }
  fclose(fp);
}

//----------------------------------------------------------------------------

void loadBad()
{
  int bad;

  FILE *fp = fopen("bad.txt", "r");
  for (int i = 0; i < dir_image_filename.size(); i++) {
    fscanf(fp, "%i: %i\n", &i, &bad);
    if (bad)
      Bad_idx_set.insert(i);
    //    printf("%i: %i\n", i, bad);
  }
  fclose(fp);

}

//----------------------------------------------------------------------------

int main( int argc, const char** argv )
{
  // splash

  printf("hello, trail!\n");

  // initialize

  trailEdgeRow.push_back(100);
  trailEdgeRow.push_back(175);

  add_all_images();

  // create initial, ordered indices 

  int i;

  for (i = 0; i < dir_image_filename.size(); i++)      
    Random_idx.push_back(i);
  printf("%i total\n", Random_idx.size());

  // shuffle indices (this should be optional)

  struct timeval tp;
  
  gettimeofday(&tp, NULL); 
  //  srand48(tp.tv_sec);
  srand(tp.tv_sec);
  random_shuffle(Random_idx.begin(), Random_idx.end());

  Nonrandom_idx.resize(Random_idx.size());
  for (i = 0; i < Random_idx.size(); i++)
    Nonrandom_idx[Random_idx[i]] = i;

  loadBad();
  printf("%i bad\n", Bad_idx_set.size());

  // display

  char c;

  do {
    
    // load image

    current_imname = dir_image_filename[current_index];

    current_im = imread(current_imname.c_str());
    draw_im = current_im.clone();

    // show image 

    draw_overlay();
    imshow("trailGT", draw_im);
    if (!callbacks_set) {
      setMouseCallback("trailGT", onMouse);
    }

    c = waitKey(0);

    onKeyPress(c);

  } while (c != (int) 'q');

  return 0;
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
