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
string fname;

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

void add_images(string dirname, vector <string> & imnames,vector<string> & filenames)
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
        filenames.push_back(string(ent->d_name));

      }
    }
    closedir (dir);
  } 
  printf("%i image files in %s\n", imnames.size(), dirname.c_str());

}
//mouse-click
bool drawing = false; //true if mouse is pressed
Mat gimg;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{

    
    if  ( event == EVENT_LBUTTONDOWN )
    {
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        drawing=true;
        circle(gimg,Point (x,y) ,6, Scalar(0,0,255),3,8,0);
        imshow("trailGT",gimg);
       

    }
    else if  ( event == EVENT_RBUTTONUP )
    {
        cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        imwrite("/Users/jiayi/Desktop/trailGT1/warthog_logs/Aug_05_2011_Fri_11_50_24_AM/omni_images_saved/"+fname, gimg);
    }
}


int main( int argc, const char** argv )
{
  printf("hello, trail!\n");
    vector<string> filenames;

  // initialize random

  struct timeval tp;
  
  gettimeofday(&tp, NULL); 
  srand48(tp.tv_sec);

  add_images(string("/warthog_logs/Aug_05_2011_Fri_11_50_24_AM/omni_images/"), dir_image_filename);
  //  add_images(imdirname, dir_image_filename);

  vector <int> collection_random_idx;
  int i=50;
    int r ;
    for (int random=0; random<dir_image_filename.size(); random++) {//use random_shuffle

        r = lrand48() % dir_image_filename.size();
        collection_random_idx.push_back(r);
    }
    
  string imname;
  string p_imname;
  string n_imname;
  int c;
  int IDX=collection_random_idx.at(i);

  do {
    fname = filenames.at(IDX);
    imname = dir_image_filename.at[IDX];
    gimg=current_im = imread(imname.c_str());
    Mat tempimg;
    imshow("trailGT", current_im);
    c = waitKey(0);
    setMouseCallback("trailGT", CallBackFunc,(void*)&gimg);
            switch (c) {
            case 110:{
            IDX=collection_random_idx.at(i++);
            n_imname = dir_image_filename.at(IDX);
            Mat next_im;
            gimg=next_im=imread(n_imname.c_str());
                imshow("trailGT",next_im);
                
                break;
            }
            case 112:
                IDX=collection_random_idx.at(i--);
                p_imname = dir_image_filename.at(IDX);
                Mat pre_im;
                gimg=pre_im=imread(p_imname.c_str());
                imshow("trailGT",pre_im);
                
                break;
            }


  } while (c != 110||112;

  return 0;
}
