#pragma once

#include "ofMain.h"
//#include "adOfxGeneral.h"
#include "ofxSpout2Sender.h"
#include "..\FrameQueue.h"
#include "ofxDelaunay.h"
#include "ofxOsc.h"
#include "ofxOpenCv.h"

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

		ofFbo overlay_fbo; // used to draw the mouth and any other overlay animations 
		ofFbo mask_fbo;

		// new queue: FaceDetectionFrame, which holds both texture and any landmark data 
		FrameQueue frame_queue;

		// calculate whether input frame is different 
		std::vector<ofFbo> difference_downsample_fbos;
		ofFbo difference_fbo;
		bool is_frame_different;
		bool checkIfFrameDifferent();
		void allocate_difference_fbos();
		int num_difference_downsamples = 5;

		void assign_camera_IDs();
		int webcam_id = 1;
		int phonecam_id = 2;
		int starting_cam = 1; // 0 webcam 1 phonecam
		void open_cam(int cam_num);

		// shaders
		ofShader shader_eye_distortion;
		ofShader shader_absdiff;
		void load_shaders();
		float difference_mode = 0;
		float difference_threshold = 0.9;

		// face detection scaling factor
		float face_detection_scalar = 0.5;
		float face_detection_crop_zoom = 1;
		
		// masks
		ofImage maskFace;

		// spout sender
		ofxSpout2::Sender spout_sender;

		// overlay mouth drawing
		ofMesh lips;
		ofMesh mouth;
		float mouth_open_amt = 0;

		// update the face mask from the pulled landmarks 
		void updateFaceMeshVertices(std::vector<ofVec2f> landmarks);
		ofxDelaunay face_triangulation;
		ofMesh face_mesh;
		ofMesh face_texture_mesh;

		// ~~~~~~
		// drawing
		// ~~~~~~
		void drawBodyPart(ofImage body_part_mask, float scale_factor, ofVec2f body_part_origin, ofVec2f body_part_destination);
		void drawBodyPart(ofImage body_part_mask, float scale_factor, ofVec2f body_part_origin);

		// osc messaging
		ofxOscReceiver osc_receiver;
		int osc_receiver_port = 7766;
		ofxOscSender osc_sender;
		int osc_sender_port = 7688;

		// check whether to record frame 
		string record_mode = "must_be_different"; // "record_all_input" "must_be_different" "must_have_face"
		bool shouldFrameBeRecorded();
		bool has_face = false;

		// opencv blob detection
		ofxCvColorImage cv_color_img;
		ofxCvGrayscaleImage cv_grey_img;
		ofxCvContourFinder cv_contour_finder;
		ofPixels cv_src_pixels;
};
