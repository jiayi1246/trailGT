#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <fstream>
#include <string>
#include<sstream>
#include<stdlib.h>
#include<math.h>


#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "dirent.h"
#include <time.h>
#include <sys/stat.h>



#include <unistd.h>
#include <sys/times.h>
#include <sys/time.h>

using namespace std;
using namespace cv;

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

void add_images(string, vector <string> &);
void add_all_images_from_file(string);
void draw_overlay();
void onMouse(int, int, int, int, void *);

void saveVert(); int loadVert();
void saveBad(); void loadBad();

int most_isolated_nonvert_image_idx();
int most_isolated_nonvert_image_idx(int);

bool isBad(int);
bool isVert(int);

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#define IMAGE_ROW_FAR    100
#define IMAGE_ROW_NEAR   175
#define ZERO_INDEX       0
#define BIG_INDEX_STEP   10
#define REQUIRED_NUMBER_OF_VERTS_PER_IMAGE   4
#define NO_INDEX         -1
#define NO_DISTANCE      -1
#define COMMENT_CHAR     '#'
#define SIMILARITY_THRESHOLD  0.05
#define MAX_DATE 12


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

vector <string> dir_image_filename;  // full-path image filenames
 
Mat current_im, draw_im;


vector <int> Random_idx;         // R[i] holds random index
vector <int> Nonrandom_idx;      // R[N[i]] = i
vector < vector <Point> > Vert;  // actual trail vertices for each image 


vector <int> ClosestVert_dist;   // for NoVert images, how many indices away is the *nearest* Vert image?
vector<int> compression_params;

                                 // max of this is "most isolated"

set <int> Bad_idx_set;           // which images are bad
set <int> Vert_idx_set;          // which images have edge vertex info
set <int> NoVert_idx_set;        // which images do NOT have edge vertex info (Vert U NoVert = all images)

set <int> FilteredVert_idx_set;  // vert images that pass various tests
set<int> resized_set;


bool do_random = false;
bool do_verts = false;
bool do_overlay = true;
bool do_bad = false;
int bad_start, bad_end;          // indices of bad range

bool bad_current_index = false;
bool vert_current_index = false;

bool callbacks_set = false;
bool dragging = false;
bool erasing = false;
int dragging_x, dragging_y;

double fontScale = 0.35;

vector <int> trailEdgeRow;

int num_saved_verts = 0;
int current_index;
string current_imname;

int max_closest_vert_dist;
int next_nonvert_idx;

bool do_show_crop_rect = false;

int output_crop_top_y = 25;
int output_crop_height = 225;   // 150
int output_crop_width = 450;  // 375

int output_width = 150;
int output_height = 75;

Mat crop_im;
Mat filtered_im;
Mat output_im(output_height,output_width,CV_8UC3);

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// smaller numbers -> MORE similar

float image_similarity(Mat & im1, Mat & im2)
{
// Calculate the L2 relative error between images
  double errorL2 = norm(im1, im2, CV_L2 );

  // Convert to a reasonable scale, since L2 error is summed across all pixels of the image
  double similarity = errorL2 / (double) ( im1.rows * im2.cols );

  return similarity;
}

//----------------------------------------------------------------------------

void filter_out_overly_similar_images()
{
  Mat im1, im2;
  float sim;
  set<int>::iterator iter, iter_next;
  
  FilteredVert_idx_set.clear();

  for (iter = Vert_idx_set.begin(); iter != Vert_idx_set.end(); iter = iter_next) {

    im1 = imread(dir_image_filename[*iter].c_str());
    //    printf("%i\n", *iter);

    FilteredVert_idx_set.insert(*iter);

    iter_next = iter;
    iter_next++;

    // find next image that is dissimilar enough

    while (iter_next != Vert_idx_set.end()) {

      im2 = imread(dir_image_filename[*iter_next].c_str());

      sim = image_similarity(im1, im2);

      if (sim < SIMILARITY_THRESHOLD) {
	printf("skipping %i [%.3f with %i]\n", *iter_next, sim, *iter);
	//	printf("BAD: %i, %i: %.3f\n", *iter, *iter_next, sim);


      }
      else {
	//	printf("GOOD: %i, %i: %.3f\n", *iter, *iter_next, sim);
	break;
      }

      iter_next++;
    }
  }

// put this in a different function which is called after filtered_*()
// don't write a file called filtered.txt -- just output the images
// call saveTrainingVert()


  printf("%i after (%i before)\n", (int) FilteredVert_idx_set.size(), (int) Vert_idx_set.size());
 
}
//----------------------------------------------------------------------------------

std::string get_date(void)
{
    time_t now = time(0);
    char buf[200];
    struct tm  tstruct;
    tstruct = *localtime(&now);

    if (now != -1)
    {
        strftime(buf, sizeof(buf), "%d_%m_%Y.%X", &tstruct);
    }
    
    return buf;
}

//-----------------------------------------------------------------------------------------------
void draw_training_images()
{
    set<int>::iterator iter, iter_next;
    //Mat filtered_im;
    //int c;
    //Mat crop_im;
    //Mat output_im(output_height,output_width,CV_8UC3);
    
    //int output_im_idx;
    string folderName = get_date().c_str();
    string folderCreateCommand = "mkdir " + folderName;
    system(folderCreateCommand.c_str());
    //std::ofstream o(folderName+"training.txt");
    
    int new_idx = 0;
    for (iter = FilteredVert_idx_set.begin(); iter != FilteredVert_idx_set.end(); iter = iter_next)

    {
//cout<<"test images"<<endl;
        filtered_im = imread(dir_image_filename[*iter].c_str());

        int x_left = filtered_im.cols / 2 - (output_crop_width/2);
        filtered_im(cv::Rect(x_left,output_crop_top_y,output_crop_width,output_crop_height)).copyTo(crop_im);
        //imwrite(dir_image_filename[*iter], crop_im);
        resize(crop_im, output_im, output_im.size(), 0, 0, INTER_LINEAR);
        //cout<<"test"<<endl;
        stringstream ss;
        string name = dir_image_filename[*iter];
        string type = ".jpg";
    
      //  ss<<folderName<<"/"<<name<<type;
        ss << folderName << "/" << new_idx++ << type;
        string fullPath = ss.str();
        cout << "writing to " << fullPath << endl;
        ss.str("");
        imwrite(fullPath, output_im);

        //resized_set.insert(*iter);
        
        imshow("crop", crop_im);
        imshow("output", output_im);
        iter_next = iter;
        iter_next++;
        
    }
        //c = waitKey(5);
}

//--------------------------------------------------------------------------------------------------
    void saveTrainingVert()
    {
  
        set<int>::iterator iter, iter_next;
        int i, j;
        float factor;
        int num_filtered_vert;
        //num_filtered_vert = FilteredVert_idx_set.size();
        //vector < vector <Point> > newVert;
        FILE *fp = fopen("training.txt", "w");
for (iter = FilteredVert_idx_set.begin(); iter != FilteredVert_idx_set.end(); iter = iter_next)
{
//num_saved_verts = 0
                filtered_im = imread(dir_image_filename[*iter].c_str());
                int x_left = filtered_im.cols / 2 - (output_crop_width/2);
                current_im(cv::Rect(x_left,output_crop_top_y,output_crop_width,output_crop_height)).copyTo(crop_im);

for (i = 0 , num_saved_verts = 0; i <Vert[*iter].size(); i++) {
            if (Vert[*iter].size() == REQUIRED_NUMBER_OF_VERTS_PER_IMAGE) {
                fprintf(fp, "%i: ", i);

                for (j = 0; j < Vert[i].size() - 1; j++)
                
                factor = (output_im.cols/crop_im.cols);
                cout<<"test vert"<<endl;
                //float r = float roundf(factor);
                int newx = ((Vert[i][j].x - x_left) * roundf(factor));
                int newy = ((Vert[i][j].y - output_crop_top_y) * roundf(factor));
 
                 //float fx = (Vert[i][j].x - x_left) * factor;
                 //newVert[i][j].x = (int) fx;
                 //float fy = (Vert[i][j].y - output_crop_top_y) * factor;
                 //newVert[i][j].y = (int) fy;
                //cout<<"test"<<endl;
                fprintf(fp, "(%i, %i), ", newx,newy);
                fprintf(fp, "(%i, %i)\n", newx,newy);
                fflush(fp);
                //num_saved_verts++;
            }
            else if (Vert[*iter].size() > 0)
            printf("improper number of vertices for image %i; not saving\n", i);
        }
}
        printf("saved %i images with verts\n",num_saved_verts);
        fclose(fp);
    
}
        
        
    
    
    







//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// change which image we are currently processing/displaying

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

  bad_current_index = isBad(current_index);

  // do we have vertices for this image?

  vert_current_index = isVert(current_index);

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

// *each line* of file loaded here should have the full path to a directory full of numbered images 

void add_all_images_from_file(string impathfilename)
{
  ifstream inStream;
  string line;

  // read file

  inStream.open(impathfilename.c_str());

  while (getline(inStream, line)) {

    cout << "loading " << line << endl;

    add_images(line, dir_image_filename);
  }


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

// what to do when mouse is moved/mouse button is push/released

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

      vert_current_index = true;

      next_nonvert_idx = most_isolated_nonvert_image_idx(current_index);
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

    vert_current_index = false;

    // this could be done more efficiently, but it should be a pretty rare event
    next_nonvert_idx = most_isolated_nonvert_image_idx();
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

// what to do when a key is pressed

void onKeyPress(char c)
{
  int idx;
  int step_idx = 1;

  // goto image 0 in sequence
  
  if (c == '0') {

    set_current_index(ZERO_INDEX);

  }

  // most isolated nonvert image remaining

  else if (c == 'n') {

    saveVert();   // for speed
    set_current_index(next_nonvert_idx);

  }

  // filter out overly-similar images

  else if (c == 'f') {

    filter_out_overly_similar_images();
    draw_training_images();
    //cout << "test"<<endl;
    saveTrainingVert();
  
    

  }

  // goto next image in sequence
  
  else if (c == 'x' || c == 'X') {

    if (c == 'X')
      step_idx = BIG_INDEX_STEP;
    
    if (!do_random) {

      if (do_verts) {
	set<int>::iterator iter = Vert_idx_set.find(current_index);
	if ((iter != Vert_idx_set.end()) && (iter == --Vert_idx_set.end()))
	  iter = Vert_idx_set.begin();
	else
	  iter++;
	set_current_index(*iter);
      }
      else
	set_current_index(current_index + step_idx);

    }
    else {
      idx = Nonrandom_idx[current_index] + step_idx;  

      if (idx >= dir_image_filename.size())
	idx = 0;

      set_current_index(Random_idx[idx]);
    }

  }
  
  // goto previous image in sequence
  
  else if (c == 'z' || c == 'Z') {
    
    if (c == 'Z')
      step_idx = BIG_INDEX_STEP;

    if (!do_random) {

      if (do_verts) {
	set<int>::iterator iter = Vert_idx_set.find(current_index);
	if (iter != Vert_idx_set.begin())
	  iter--;
	else {
	  iter = Vert_idx_set.end();
	  --iter;
	}
	set_current_index(*iter);
	
      }
      else
	set_current_index(current_index - step_idx);
      
    }
    else {
      idx = Nonrandom_idx[current_index] - step_idx;  

      if (idx < 0)
	idx = dir_image_filename.size() - 1;

      set_current_index(Random_idx[idx]);
    }   
  }
  
  // toggle showing crop rectangle

  else if (c == 'c') 
    do_show_crop_rect = !do_show_crop_rect;

  // toggle overlay
  
  else if (c == 'o') 
    do_overlay = !do_overlay;
  
  // toggle randomized index mode
  
  else if (c == 'r') {
    if (!do_verts)
      do_random = !do_random;
  }

  // toggle vert image-only mode
  
  else if (c == 'v') {
    if (vert_current_index) {
      do_verts = !do_verts;
      if (do_verts)
	do_random = false;
    }
  }

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

  // save everything

  else if (c == 's') {
    saveVert();
    saveBad();
    saveTrainingVert();
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

void draw_output_window()
{
   int c; 
   Mat crop_im;
   Mat output_im(output_height,output_width,CV_8UC3);

   int x_left = current_im.cols / 2 - (output_crop_width/2);
   current_im(cv::Rect(x_left,output_crop_top_y,output_crop_width,output_crop_height)).copyTo(crop_im);

/*int output_crop_top_y = 25;
int output_crop_height = 225;   // 150
int output_crop_width = 450;  // 375

int output_width = 150;
int output_height = 75;*/

   resize(crop_im, output_im, output_im.size(), 0, 0, INTER_LINEAR);

   imshow("crop", crop_im);
   imshow("output", output_im);

    c = waitKey(5);
}

// draw strings and shapes on top of current image

void draw_overlay()
{
  stringstream ss;

  if (do_overlay) {
    
    // which image is this?

    ss << current_index << ": " << current_imname;
    string str = ss.str();

    putText(draw_im, str, Point(5, 10), FONT_HERSHEY_SIMPLEX, fontScale, Scalar::all(255), 1, 8);

    // isolation stats

    if (!bad_current_index && !vert_current_index) {
      ss.str("");
      ss << "max dist = " << max_closest_vert_dist << ", this dist = " << ClosestVert_dist[current_index];
      putText(draw_im, ss.str(), Point(5, 25), FONT_HERSHEY_SIMPLEX, fontScale, Scalar(255, 255, 255), 1, 8);
    }

    // save status

    int num_unsaved = Vert_idx_set.size() - num_saved_verts;

    if (num_unsaved > 0) {
      ss.str("");
      if (num_unsaved == 1)
	ss << num_unsaved << " image with unsaved verts [" << Vert_idx_set.size() << "]";
      else
	ss << num_unsaved << " images with unsaved verts [" << Vert_idx_set.size() << "]";
      str = ss.str();
      putText(draw_im, str, Point(5, 315), FONT_HERSHEY_SIMPLEX, 1.5 * fontScale, Scalar(0, 0, 255), 1, 8);
    }

    // show crop rectangle:

    if (do_show_crop_rect) {

      int center_x = draw_im.cols / 2;
      int xl = center_x - output_crop_width/2;
      int xr = center_x + output_crop_width/2;
      int yt = output_crop_top_y;
      int yb = output_crop_top_y + output_crop_height;

      rectangle(draw_im,  
		Point(xl, yt),
		Point(xr, yb),
		Scalar(255, 255, 255), 2);
      
    }

    // are we in "random next image" mode?

    if (do_random) 
      putText(draw_im, "R", Point(5, 40), FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 255), 1, 8);

    // are we in "bad image" marking mode?

    if (do_bad) 
      putText(draw_im, "B", Point(15, 40), FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 255), 1, 8);

    // are we in "verts only" mode?

    if (do_verts) 
      putText(draw_im, "V", Point(25, 40), FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 255), 1, 8);

    // is this a "bad" image?

    if (bad_current_index) {
      setChannel(draw_im, 2, 255);
      return;
    }

    // is this an image for which we have ground-truth trail edges?

    else if (vert_current_index) 
      setChannel(draw_im, 0, 200);
   
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

    for (int i = 0; i < Vert[current_index].size(); i += 2) {

      // only draw line segment if we have a PAIR of verts (this will NOT be the case while a new segment is being drawn) 

      if (i + 1 < Vert[current_index].size())
	line(draw_im, Vert[current_index][i], Vert[current_index][i + 1], Scalar(0, g, r), 2);
    }

  }
}

//----------------------------------------------------------------------------

// write which images have been marked as "bad" -- i.e., they contain no trail

void saveBad()
{
  FILE *fp = fopen("bad.txt", "w");
  for (int i = 0; i < dir_image_filename.size(); i++) {
    fprintf(fp, "%i: %i\n", i, isBad(i));
    fflush(fp);
  }
  fclose(fp);
}

//----------------------------------------------------------------------------

// read file that specifies which images should be ignored because they contain
// no trail or the trail geometry does not conform to assumptions

// this assumes image filenames have already been loaded, so we know how many lines
// should be in this file

void loadBad()
{
  int bad;

  FILE *fp = fopen("bad.txt", "r");
  for (int i = 0; i < dir_image_filename.size(); i++) {
    fscanf(fp, "%i: %i\n", &i, &bad);
    if (bad)
      Bad_idx_set.insert(i);
  }
  fclose(fp);

}

//----------------------------------------------------------------------------

// check whether a given image is contained in current set of "bad" (non-trail") images

bool isBad(int idx)
{
  return (Bad_idx_set.find(idx) != Bad_idx_set.end());
}

//----------------------------------------------------------------------------

// check whether a given image is contained in set of those with ground-truth vertices
// specified

bool isVert(int idx)
{
  return (Vert_idx_set.find(idx) != Vert_idx_set.end());
}

//----------------------------------------------------------------------------

// write all current ground-truth trail vertices to file

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
	 (int) Vert_idx_set.size(),
	 (int) NoVert_idx_set.size(),
	 (int) Vert.size());
  fclose(fp);
}

//----------------------------------------------------------------------------

// utility function for reading ground-truth vertices from file

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

// utility function for reading ground-truth coordinate pair (x, y) from file

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

// how many "images away" from this one is the nearest image for which we have ground-truth?

// i should be the index of a nonvert image

void calculate_closest_vert_dist(int i)
{
  int j, last_dist, next_dist;

  // get NEXT vert index
  
  for (j = i + 1, next_dist = NO_DISTANCE; j < Vert.size() && (isBad(j) || !isVert(j)); j++)
    ;
  
  if (j >= 0 && j < Vert.size() && isVert(j)) 
    next_dist = j - i;
  
  // get LAST vert index 
  
  for (j = i - 1, last_dist = NO_DISTANCE; j >= 0 && (isBad(j) || !isVert(j)); j--)
    ;
  if (j >= 0 && j < Vert.size() && isVert(j)) 
    last_dist = i - j;
  
  // is forward or backward dist smaller?
  
  // at least one distance is "infinite"
  
  if (next_dist == NO_DISTANCE) 
    ClosestVert_dist[i] = last_dist;
  else if (last_dist == NO_DISTANCE) 
    ClosestVert_dist[i] = next_dist;
  
  // both distances normal
  
  else {
    if (next_dist <= last_dist) 
      ClosestVert_dist[i] = next_dist;
    else 
      ClosestVert_dist[i] = last_dist;
  }
}

//----------------------------------------------------------------------------

// a new vert has been inserted.  recalculate MINIMUM number of distances from 
// non-vert images to nearest vert image

int most_isolated_nonvert_image_idx(int new_vert_idx)
{
  int i, j, max_dist, max_dist_idx;

  // inserted vertex is now out of the game

  ClosestVert_dist[new_vert_idx] = NO_DISTANCE;

  // forward non-verts

  for (j = new_vert_idx + 1; j < Vert.size() && !isVert(j); j++)
    if (!isBad(j)) 
      calculate_closest_vert_dist(j);

  // backward non-verts

  for (j = new_vert_idx - 1; j >= 0 && !isVert(j); j--)
    if (!isBad(j)) 
      calculate_closest_vert_dist(j);

  // now find global max

  for (i = 0, max_dist = 0, max_dist_idx = NO_INDEX; i < Vert.size(); i++) 
    if (ClosestVert_dist[i] != NO_DISTANCE && ClosestVert_dist[i] > max_dist) {
      max_dist = ClosestVert_dist[i];
      max_dist_idx = i;
    }

  max_closest_vert_dist = max_dist;

  return max_dist_idx;
}


//----------------------------------------------------------------------------

// calculate distances from ALL non-vert images to nearest vert image.  
// expensive, but should be called only once, when program is initialized

int most_isolated_nonvert_image_idx()
{
  int i, j, last_dist, next_dist, max_dist, max_dist_idx;

  
  for (i = 0; i < Vert.size(); i++) {
    
    if (!isBad(i) && !isVert(i)) 
      calculate_closest_vert_dist(i);
    else
      ClosestVert_dist[i] = NO_DISTANCE;
  }

  for (i = 0, max_dist = 0, max_dist_idx = NO_INDEX; i < Vert.size(); i++) 
    if (ClosestVert_dist[i] != NO_DISTANCE && ClosestVert_dist[i] > max_dist) {
      max_dist = ClosestVert_dist[i];
      max_dist_idx = i;
    }

  max_closest_vert_dist = max_dist;

  return max_dist_idx;
}

//----------------------------------------------------------------------------

// get all ground-truth trail vertices from file

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

    // skip comments

    if (line[0] == COMMENT_CHAR)
      continue;

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

    // get x, y coords of vert

    while (getNextVert(line, line_idx, line_idx, x_val, y_val)) {

      v.x = x_val;
      v.y = y_val;
      Vert[image_idx].push_back(v);
    }

    num_verts++;
  }
 
  inStream.close();

  next_nonvert_idx = most_isolated_nonvert_image_idx();

  return num_verts;
}



//----------------------------------------------------------------------------

int main( int argc, const char** argv )
{
  // splash

  printf("hello, trail!\n");

  // initialize for LINEAR trail approximation (quadrilateral)

  trailEdgeRow.push_back(IMAGE_ROW_FAR);
  trailEdgeRow.push_back(IMAGE_ROW_NEAR);

  add_all_images_from_file("imagedirs.txt");

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
  ClosestVert_dist.resize(Random_idx.size());   

  num_saved_verts = loadVert();
  printf("%i with verts\n", num_saved_verts);

  set_current_index(ZERO_INDEX);

  // display

  char c;

  do {
    
    // load image

    current_imname = dir_image_filename[current_index];

    current_im = imread(current_imname.c_str());
    draw_im = current_im.clone();

    // show image 

    draw_overlay();
    //draw_output_window();
    //draw_training_images();


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
