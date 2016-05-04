#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>

#include "dirent.h"

#include <unistd.h>
#include <sys/times.h>
#include <sys/time.h>

using namespace std;
using namespace cv;

// put this in GitHub

string imdirname = "/warthog_logs/Aug_05_2011_Fri_11_50_24_AM/omni_images/";

vector <string> dir_image_filename;  // full-path image filenames
//vector <GT *> image_GT;   
 
Mat current_im;

// if being run for first time: 
//  -- go through directories and gather image names
//  -- write some kinda database file with names, "empty" GT info (serialize map with OpenCV's XML output capabilities?) 
// if database file exists, read it and print info (how many images, how many have GT)

// pick a random image (subject to constraints, like not too close to existing choice) and show it
// UI for picking point(s) [such as constraint on Y val], saving to GT

// mode for showing existing GT points
// save key to re-write entire database

void add_images(string dirname, vector <string> & imnames)
{
  string imname;

  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir (dirname.c_str())) != NULL) {
    /* print all the JPEGs/PNGs within the directory */
    while ((ent = readdir (dir)) != NULL) {
      if (strstr(ent->d_name, "jpg") != NULL || strstr(ent->d_name, "png") != NULL) {
	
	imname = dirname + string(ent->d_name);
	imnames.push_back(imname);

      }
    }
    closedir (dir);
  } 
  printf("%i image files in %s\n", imnames.size(), dirname.c_str());

}

int main( int argc, const char** argv )
{
  printf("hello, trail!\n");

  // initialize random

  struct timeval tp;
  
  gettimeofday(&tp, NULL); 
  srand48(tp.tv_sec);

  add_images(string("/warthog_logs/Aug_05_2011_Fri_11_50_24_AM/omni_images/"), dir_image_filename);
  //  add_images(imdirname, dir_image_filename);

  
  string imname;
  int c;

  do {

    int r = lrand48() % dir_image_filename.size();

    imname = dir_image_filename[r];

    current_im = imread(imname.c_str());

    imshow("trailGT", current_im);
    c = waitKey(0);

  } while (c != (int) 'q');

  return 0;
}
