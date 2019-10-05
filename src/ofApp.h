#pragma once

#include "ofMain.h"
//#include "adOfxGeneral.h"
#include "ofxSpout2Sender.h"
#include "..\FrameQueue.h"

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		
		//adImageSequenceRecorder seq_recorder;
		ofVideoGrabber cam_grabber;
		ofFbo output_fbo;

		// camera texture queue 
		ofTexture selected_texture; // either the camera input or the queue, depending on what's picked 
		int texture_queue_size = 24;
		std::vector<ofFbo> cam_tex_queue;
		void allocate_cam_tex_queue();
		ofFbo cam_cropped;
		void crop_cam();
		int previous_tex_queue_index = -1; // the index of the last frame shown from the queue 
		int last_stored_tex_queue_index = -1; // the index of the last updated texture queue frame 
		int total_queued_textures = 0; // total number of times we've stored a texture into the queue

		// new queue: FaceDetectionFrame, which holds both texture and any landmark data 
		FrameQueue frame_queue;

		// calculate whether input frame is different 
		std::vector<ofFbo> difference_downsample_fbos;
		ofFbo difference_fbo;
		bool is_frame_different;
		bool checkIfFrameDifferent();
		void allocate_difference_fbos();
		int num_difference_downsamples = 5;

		void open_cam(int cam_num);

		// shaders
		ofShader shader_eye_distortion;
		ofShader shader_absdiff;
		void load_shaders();

		// face detection scaling factor
		float face_detection_scalar = 0.5;
		float face_detection_crop_zoom = 1;
		
		// masks
		ofImage maskFace;

		// spout sender
		ofxSpout2::Sender spout_sender;
};
