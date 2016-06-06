#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <fstream>
#include <string>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

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

void saveVert(); int loadVert();
void saveBad(); void loadBad();

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#define BIG_INDEX_STEP   10
#define REQUIRED_NUMBER_OF_VERTS_PER_IMAGE   4

string imdirname = "/warthog_logs/Aug_05_2011_Fri_11_50_24_AM/omni_images/";
string fname;

vector <string> dir_image_filename;  // full-path image filenames
 
Mat current_im, draw_im;

vector <int> Random_idx;         // R[i] holds random index
vector <int> Nonrandom_idx;      // R[N[i]] = i
vector < vector <Point> > Vert;  // actual trail vertices for each image 

set <int> Bad_idx_set;           // which images are bad
set <int> Vert_idx_set;          // which images have edge vertex info
set <int> NoVert_idx_set;        // which images do NOT have edge vertex info (Vert U NoVert = all images)

bool do_random = false;
bool do_overlay = true;
bool do_bad = false;
int bad_start, bad_end;          // indices of bad range

bool bad_current_index = false;

bool callbacks_set = false;
bool dragging = false;
bool erasing = false;
int dragging_x, dragging_y;

double fontScale = 0.35;

vector <int> trailEdgeRow;

int num_saved_verts = 0;
int current_index = 0;
string current_imname;

//----------------------------------------------------------------------------

// if being run for first time: 
//  -- go through directories and gather image names
//  -- write some kinda database file with names, "empty" GT info (serialize map with OpenCV's XML output capabilities?) 
// if database file exists, read it and print info (how many images, how many have GT)

//----------------------------------------------------------------------------

void set_current_index(int new_index)
{
  int n = dir_image_filename.size();

  // if we changed image index mid-way through choosing vertices, CLEAR those vertices

  if (Vert[current_index].size() > 0 && Vert[current_index].size() != REQUIRED_NUMBER_OF_VERTS_PER_IMAGE) {
    Vert[current_index].clear();
  }

  // wrap?

  if (new_index >= n)
    current_index = n - new_index;
  else if (new_index < 0)
    current_index = n + new_index;

  // normal

  else
    current_index = new_index;

  // is this a "bad" image?

  bad_current_index = (Bad_idx_set.find(current_index) != Bad_idx_set.end());  
}

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
  Point v;
  int g, b;

  // no mouse interaction if image being or already marked bad

  if (do_bad || bad_current_index) 
    return;

  // dragging...

  if  ( event == EVENT_MOUSEMOVE ) {

    //    vx = x;
    v.x = x;
    if (dragging) {
      g = 200;
      b = 0;
      v.y = dragging_y;
    }
    else {
      g = 255; 
      b = 255;
      v.y = snap_y(y);
    }
  }

  // initiating horizontal segment

  else if  ( event == EVENT_LBUTTONDOWN ) {

    dragging = true;

    g = 255;
    b = 0;
    v.x = x;
    v.y = snap_y(y);
    dragging_x = v.x;
    dragging_y = v.y;

    // clear any existing verts *on this row*

    vector <Point>::iterator iter = Vert[current_index].begin();

    while (iter != Vert[current_index].end()) {
      if ((*iter).y == v.y)
	iter = Vert[current_index].erase(iter);
      else
	iter++;
    }

    Vert[current_index].push_back(v);

  }

  // finishing horizontal segment

  else if  ( event == EVENT_LBUTTONUP ) {

    dragging = false;

    g = 255;
    b = 0;
    v.x = x;
    v.y = dragging_y;

    Vert[current_index].push_back(v);

    // done with this image!

    if (Vert[current_index].size() == REQUIRED_NUMBER_OF_VERTS_PER_IMAGE) {

      Vert_idx_set.insert(current_index);

      set<int>::iterator iter = NoVert_idx_set.find(current_index);
      // if it is actually in the set, erase it
      if (iter != NoVert_idx_set.end())
	NoVert_idx_set.erase(iter);

    }

  }

  // clear vertices for this image

  else if  ( event == EVENT_RBUTTONDOWN ) {

    erasing = true;

  }

  // clear vertices for this image

  else if  ( event == EVENT_RBUTTONUP ) {

    erasing = false;

    Vert[current_index].clear();

    set<int>::iterator iter = Vert_idx_set.find(current_index);
    // if it is actually in the set, erase it
    if (iter != Vert_idx_set.end())
      Vert_idx_set.erase(iter);

    NoVert_idx_set.insert(current_index);
  }

  draw_im = current_im.clone();
  draw_overlay();

  // show current edit that is underway

  if (dragging)
    line(draw_im, Point(dragging_x, v.y), Point(v.x, v.y), Scalar(0, 255, 255), 2);

  // where is cursor?

  if (!erasing)
    circle(draw_im, Point (v.x, v.y), 8, Scalar(0, g, b), 1, 8, 0);

  imshow("trailGT", draw_im);  
}

//----------------------------------------------------------------------------

void onKeyPress(char c)
{
  int idx;
  int step_idx = 1;

  // goto image 0 in sequence
  
  if (c == '0') {

    set_current_index(0);

  }

  // goto next image in sequence
  
  else if (c == 'x' || c == 'X') {

    if (c == 'X')
      step_idx = BIG_INDEX_STEP;
    
    if (!do_random) {

      set_current_index(current_index + step_idx);

      /*
      current_index += step_idx;  
      
      // wrap?
      
      if (current_index >= dir_image_filename.size())
	current_index = 0;
      */
    }
    else {
      idx = Nonrandom_idx[current_index] + step_idx;  

      if (idx >= dir_image_filename.size())
	idx = 0;

      //      current_index = Random_idx[idx];
      set_current_index(Random_idx[idx]);
    }

  }
  
  // goto previous image in sequence
  
  else if (c == 'z' || c == 'Z') {
    
    if (c == 'Z')
      step_idx = BIG_INDEX_STEP;

    if (!do_random) {

      set_current_index(current_index - step_idx);

      /*
      current_index -= step_idx;
      
      // wrap?
      
      if (current_index < 0)
	current_index = dir_image_filename.size() - 1;
      */
    }
    else {
      idx = Nonrandom_idx[current_index] - step_idx;  

      if (idx < 0)
	idx = dir_image_filename.size() - 1;

      //      current_index = Random_idx[idx];
      set_current_index(Random_idx[idx]);
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
    bad_current_index = false;
  }

  // save

  else if (c == 's') {
    saveVert();
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

    int num_unsaved = Vert_idx_set.size() - num_saved_verts;

    if (num_unsaved > 0) {
      ss.str("");
      if (num_unsaved == 1)
	ss << num_unsaved << " image with unsaved verts";
      else
	ss << num_unsaved << " images with unsaved verts";
      str = ss.str();
      putText(draw_im, str, Point(5, 315), FONT_HERSHEY_SIMPLEX, 1.5 * fontScale, Scalar(0, 0, 255), 1, 8);
    }

    // are we in "random next image" mode?

    if (do_random) 
      putText(draw_im, "R", Point(5, 25), FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 255), 1, 8);

    // are we in "bad image" marking mode?

    if (do_bad) 
      putText(draw_im, "B", Point(15, 25), FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 255), 1, 8);

    // is *this* "bad" image?

    if (bad_current_index) {
      setChannel(draw_im, 2, 255);
      return;
    }

    // horizontal lines for trail edge rows

    for (int i = 0; i < trailEdgeRow.size(); i++) 
      line(draw_im, Point(0, trailEdgeRow[i]), Point(current_im.cols - 1, trailEdgeRow[i]), Scalar(0, 128, 128), 1);

    // trail edge vertices

    int r, g;

    if (erasing) {
      r = 255; g = 0;
    }
    else {
      r = 0; g = 255;
    }

    //    if (!(Vert[current_index].size() % 2)) {
    for (int i = 0; i < Vert[current_index].size(); i += 2) {

      // only draw line segment if we have a PAIR of verts (this will NOT be the case while a new segment is being drawn) 
      if (i + 1 < Vert[current_index].size())
	line(draw_im, Vert[current_index][i], Vert[current_index][i + 1], Scalar(0, g, r), 2);
    }

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

// requires that an image have exactly 4 verts to be written

void saveVert()
{
  int i, j;

  FILE *fp = fopen("vert.txt", "w");

  for (i = 0, num_saved_verts = 0; i < Vert.size(); i++) {
    if (Vert[i].size() == REQUIRED_NUMBER_OF_VERTS_PER_IMAGE) {
      fprintf(fp, "%i: ", i);
      for (j = 0; j < Vert[i].size() - 1; j++)
	fprintf(fp, "(%i, %i), ", Vert[i][j].x, Vert[i][j].y);
      fprintf(fp, "(%i, %i)\n", Vert[i][j].x, Vert[i][j].y);
      fflush(fp);
      num_saved_verts++;
    }
    else if (Vert[i].size() > 0)
      printf("improper number of vertices for image %i; not saving\n", i);
  }
  printf("saved %i images with verts [%i S + %i U = %i]\n", 
	 num_saved_verts, 
	 Vert_idx_set.size(),
	 NoVert_idx_set.size(),
	 Vert.size());
  fclose(fp);
}

//----------------------------------------------------------------------------

// extract next integer from string starting from index startPos
// if s[startPos] is NOT a digit, keeps searching until one is found
// if no digit found before end of string, error

bool getNextInt(string s, int startPos, int & curPos, int & nextInt)
{
  string s_nextint = "";

  for (curPos = startPos; curPos < s.length() && (isdigit(s[curPos]) || s_nextint.length() == 0); curPos++) {
    if (isdigit(s[curPos])) 
      s_nextint += s[curPos];
    //    cout << curPos << " " << s_nextint << endl;

  }

  if (s_nextint.length() > 0) {
    nextInt = atoi(s_nextint.c_str());
    return true;
  }
  else
    return false;

}

//----------------------------------------------------------------------------

bool getNextVert(string s, int startPos, int & curPos, int & x, int & y)
{
  // get x coord

  if (!getNextInt(s, startPos, curPos, x)) 
    return false;

  // get y coord

  if (!getNextInt(s, curPos, curPos, y)) {
    printf("loadVert(): x coordinate but no y! [%s]\n", s.c_str());
    exit(1);
  }

  return true;
}

//----------------------------------------------------------------------------

// assumes Vert vector is already the right size, so that vertices may be added
// at the appropriate indices

int loadVert()
{
  ifstream inStream;
  string line;
  int num_verts = 0;
  int i, line_idx, image_idx, x_val, y_val;
  Point v;

  // initialize NoVert_idx_set to all images

  for (i = 0; i < Vert.size(); i++)
    NoVert_idx_set.insert(i);

  // read file

  inStream.open("vert.txt");

  while (getline(inStream, line)) {

    // get image index

    if (!getNextInt(line, 0, line_idx, image_idx)) {
      printf("loadVert(): problem parsing index on line %i\n", num_verts);
      exit(1);
    }

    Vert_idx_set.insert(image_idx);

    set<int>::iterator iter = NoVert_idx_set.find(image_idx);
    // if it is actually in the set, erase it
    if (iter != NoVert_idx_set.end())
      NoVert_idx_set.erase(iter);

    //    printf("index %i (%i)\n", image_idx, line_idx);

    // get x, y coords of vert

    while (getNextVert(line, line_idx, line_idx, x_val, y_val)) {

      v.x = x_val;
      v.y = y_val;
      Vert[image_idx].push_back(v);

      //      printf("  (%i, %i)\n", x_val, y_val);
    }

    num_verts++;
  }
 

  inStream.close();

  return num_verts;
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
  printf("%i total images\n", (int) Random_idx.size());

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
  printf("%i bad\n", (int) Bad_idx_set.size());

  Vert.resize(Random_idx.size());

  num_saved_verts = loadVert();
  printf("%i with verts\n", num_saved_verts);

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
