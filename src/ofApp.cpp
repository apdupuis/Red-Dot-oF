                                                                                                                                                     #include "ofApp.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

//#include "opencv2/opencv.hpp"

#include "ofxSpout2Receiver.h"

using namespace dlib;
using namespace std;
//using namespace cv;

// ~~~~~~~~~~~~~~~~
// GLOBAL OBJECTS
// ~~~~~~~~~~~~~~~~

// CAMERA PARAMS 

int WEBCAM = 0;
int PHONECAM = 1;
int current_cam = WEBCAM;

ofVec2f cam_size;
ofVec2f cam_output_size;
ofVec4f cam_bounds;

ofVec2f webcam_size = ofVec2f(640, 360);
ofVec4f webcam_bounds = ofVec4f(0, 0, 640, 360);
ofVec2f phonecam_size = ofVec2f(448, 800);
ofVec4f phonecam_bounds = ofVec4f(0, 0, 448, 615);

int get_cam_width() {
	return cam_output_size.x;
}

int get_cam_height() {
	return cam_output_size.y;
}

void ofApp::allocate_cam_tex_queue()
{
	cam_tex_queue.clear();

	for (int i = 0; i < texture_queue_size; i++) {
		ofFbo cam_tex_temp;
		cam_tex_temp.allocate(get_cam_width(), get_cam_height(), GL_RGBA, 1);
		cam_tex_queue.push_back(cam_tex_temp);
	}
}

void ofApp::crop_cam()
{
	cam_cropped.begin();
	cam_grabber.getTexture().drawSubsection(0, 0, get_cam_width(), get_cam_height(), cam_bounds.x, cam_bounds.y, cam_bounds.z, cam_bounds.w);
	cam_cropped.end();
}

bool ofApp::checkIfFrameDifferent()
{
	// if we haven't stored a texture yet it should by definition be different 
	if (total_queued_textures == 0) {
		return true;
	}

	ofFbo previous_stored_tex = cam_tex_queue[last_stored_tex_queue_index];
	//previous_stored_tex.draw(get_cam_width(), 0);
	
	difference_fbo.begin();
	shader_absdiff.begin();
	shader_absdiff.setUniformTexture("tex0", cam_grabber.getTextureReference(), 0);
	shader_absdiff.setUniformTexture("tex1", previous_stored_tex.getTextureReference(), 1);
	cam_grabber.getTexture().draw(0, 0);
	shader_absdiff.end();
	difference_fbo.end();

	difference_fbo.draw(get_cam_width(), 0);

	if (difference_downsample_fbos.size() > 0) {
		ofFbo initial_downsample = difference_downsample_fbos[0];
		initial_downsample.begin();
		difference_fbo.draw(0, 0, initial_downsample.getWidth(), initial_downsample.getHeight());
		initial_downsample.end();

		for (int i = 0; i < difference_downsample_fbos.size() - 1; i++) {
			ofFbo src_fbo = difference_downsample_fbos[i];
			ofFbo dst_fbo = difference_downsample_fbos[i+1];

			dst_fbo.begin();
			src_fbo.draw(0, 0, dst_fbo.getWidth(), dst_fbo.getHeight());
			dst_fbo.end();
			dst_fbo.draw(get_cam_width(), get_cam_height());
		}

		ofFbo one_by_one = difference_downsample_fbos[difference_downsample_fbos.size() - 1];
		ofPixels diff_pixels;
		one_by_one.readToPixels(diff_pixels);
		ofColor diff_color = diff_pixels.getColor(0);
		if (diff_color.r > 0 || diff_color.g > 0 || diff_color.b > 0) {
			return true;
		}
		else {
			return false;
		}
	}

	return false;
}

void ofApp::allocate_difference_fbos()
{
	for (int i = num_difference_downsamples; i > 0; i--) {
		ofFbo difference_fbo_temp;
		int tex_width = pow(2, i - 1);
		difference_fbo_temp.allocate(tex_width, tex_width, GL_RGBA, 1);
		difference_downsample_fbos.push_back(difference_fbo_temp);
		cout << "allocated difference fbo with dim " << difference_fbo_temp.getWidth() << " " << difference_fbo_temp.getHeight() << endl;
	}
}

void ofApp::assign_camera_IDs()
{
	std::vector<ofVideoDevice> vid_devices = cam_grabber.listDevices();
	for (int i = 0; i < vid_devices.size(); i++) {
		string camera_name = vid_devices[i].deviceName;
		if (camera_name == "USB Camera") {
			webcam_id = i;
			cout << "Found webcam" << endl;
		}
		if (camera_name == "OBS-Camera") {
			phonecam_id = i;
			cout << "Found phonecam" << endl;
 		}
	}
}

void ofApp::open_cam(int cam_num) {

	if (cam_grabber.isInitialized()) {
		cam_grabber.close();
	}

	switch (cam_num) {
	case 0:
		cam_grabber.setDeviceID(webcam_id);
		cam_size = webcam_size;
		cam_bounds = webcam_bounds;
		break;
	case 1:
		cam_grabber.setDeviceID(phonecam_id);
		cam_size = phonecam_size;
		cam_bounds = phonecam_bounds;
		break;
	default:
		break;
	}

	cam_output_size = ofVec2f(cam_bounds.z - cam_bounds.x, cam_bounds.w - cam_bounds.y);
	cam_grabber.setup(cam_size.x, cam_size.y);
	output_fbo.allocate(get_cam_width(), get_cam_height(), GL_RGBA, 4);

	// initialize camera crop fbo
	cam_cropped.allocate(get_cam_width(), get_cam_height(), GL_RGBA, 1);
	// initialize fbo for calculating difference
	difference_fbo.allocate(get_cam_width(), get_cam_height(), GL_RGBA, 1);
	// initialize fbo for calculating difference between frames
	difference_fbo.allocate(get_cam_width(), get_cam_height(), GL_RGBA, 1);
	// initialize texture queue
	allocate_cam_tex_queue();

	// initialize the overlay fbo 
	overlay_fbo.allocate(get_cam_width(), get_cam_height(), GL_RGBA, 1);
	mask_fbo.allocate(get_cam_width(), get_cam_height(), GL_RGBA, 1);

	allocate_difference_fbos();
	//frame_queue.setup(get_cam_width(), get_cam_height(), texture_queue_size);
}

void ofApp::load_shaders()
{
	shader_eye_distortion.load("passthrudim.vert", "ad.eye_distortion.frag");
	shader_absdiff.load("passthrudim.vert", "ad.is_different.frag");
}

void ofApp::updateFaceMeshVertices(std::vector<ofVec2f> landmarks)
{
	// reset the triangulation in case we want to store it for a given frame 
	face_triangulation.reset();

	for (int k = 0; k < landmarks.size(); k++) {
		ofVec2f current_point = landmarks[k];

		// add to triangulation
		face_triangulation.addPoint(ofPoint(current_point));

		// add to the face_mesh if we want to draw it with these coordinates 
		face_mesh.setVertex(k, ofPoint(current_point));

		// add to the face texture mesh as texture coordinate 
		face_texture_mesh.setTexCoord(k, current_point);
	}
}

// dlib compatible image from camera
array2d<rgb_pixel> camera_img_dlib_compatible;

// the index of the currently highlighted landmark, used for determining the various landmark indices 
int current_highlighted_landmark = 0;

// list of detected faces
std::vector<dlib::rectangle> detected_faces;

// the face detector 
frontal_face_detector face_detector = get_frontal_face_detector();

// we'll use this vector to store face landmarks in oF screen space  
std::vector<ofVec2f> face_landmarks;

// an fbo containing the most recently captured face, in mask space 
ofFbo face_texture_fbo;

// list of texture coordinates for the face mask
std::vector<ofVec2f> face_texcoords;

std::vector<ofFbo> fbo_sequence;

// set of 6 face points calculated by dlib, used to calculate head pose 
//std::vector<cv::Point2d> head_orientation_points;

// points for a platonic face model, used for estimating head orientation
//std::vector<cv::Point3d> head_model_points;

// masks for face parts
ofImage maskEyeRight, maskEyeLeft, maskMouth, maskNose;

// nose end points for head pose estimation: probably don't need to be global 
//std::vector<cv::Point3d> nose_end_point3D;
//std::vector<cv::Point2d> nose_end_point2D;

// previous camera shot
ofFbo previous_cam_tex;

// record current frame's detection points, triangulation, and face texture 
bool record_face_alignment = false;

// used for getting face landmarks from an image (camera_img_dlib_compatible) and a list of detected faces (detected_faces)
shape_predictor sp;

// spout receiver from max 
//ofxSpout2::Receiver spout_receiver_max1;

// translation vector for head pose 
//cv::Mat translation_vector;


// ~~~~~~~~~~~~~~~~
// FUNCTIONS 
// ~~~~~~~~~~~~~~~~

// draw body part overlaid 

void ofApp::drawBodyPart(ofImage body_part_mask, float scale_factor, ofVec2f body_part_origin, ofVec2f body_part_destination) {
	face_texture_fbo.getTexture().setAlphaMask(body_part_mask.getTexture());
	ofPushMatrix();
	ofTranslate(body_part_destination - body_part_origin);
	ofTranslate(body_part_origin);
	ofScale(scale_factor);
	ofTranslate(-body_part_origin);
	glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	face_texture_fbo.getTexture().bind();
	face_mesh.draw();
	face_texture_fbo.getTexture().unbind();
	ofEnableAlphaBlending();
	ofPopMatrix();
}

void ofApp::drawBodyPart(ofImage body_part_mask, float scale_factor, ofVec2f body_part_origin) {
	drawBodyPart(body_part_mask, scale_factor, body_part_origin, body_part_origin);
}

bool ofApp::shouldFrameBeRecorded()
{
	if (record_mode == "record_all_input") 
	{
		return true;
	}
	if (record_mode == "must_be_different")
	{
		return is_frame_different;
	}
	if (record_mode == "must_have_face")
	{
		return is_frame_different && has_face;
	}
	else
	{
		return true;
	}
}

// returns head orientation point in cv format at a given index from a list of dlib landmarks
//cv::Point2d extract_head_orientation_point(std::vector<ofVec2f> face_landmarks_vector, int index) {
//	float shape_x = face_landmarks_vector[index].x;
//	float shape_y = face_landmarks_vector[index].y;
//	return cv::Point2d(shape_x, shape_y);
//}
//
//cv::Point2d extract_head_orientation_point_translated(std::vector<ofVec2f> face_landmarks_vector, int index) {
//	cv::Point2d pt = extract_head_orientation_point(face_landmarks_vector, index);
//	return pt - cv::Point2d(get_cam_width() / 2, get_cam_height() / 2);
//}

// extracts all head orientation points used for head pose estimation 
//std::vector<cv::Point2d> get_head_orientation_points(std::vector<ofVec2f> face_landmarks_vector) {
//	std::vector<cv::Point2d> head_orientation_points_temp;
//	head_orientation_points_temp.push_back(extract_head_orientation_point(face_landmarks_vector, 30));	// Nose tip
//	head_orientation_points_temp.push_back(extract_head_orientation_point(face_landmarks_vector, 8));	// Chin
//	head_orientation_points_temp.push_back(extract_head_orientation_point(face_landmarks_vector, 45));	// Left eye left corner 
//	head_orientation_points_temp.push_back(extract_head_orientation_point(face_landmarks_vector, 36));	// Right eye right corner
//	head_orientation_points_temp.push_back(extract_head_orientation_point(face_landmarks_vector, 54));	// Left Mouth corner
//	head_orientation_points_temp.push_back(extract_head_orientation_point(face_landmarks_vector, 48));	// Right mouth corner
//	return head_orientation_points_temp;
//}

// mouth center in screen space 
ofVec2f getMouthCenter(std::vector<ofVec2f> face_landmarks_vector) {
	ofVec2f left_mouth_corner = face_landmarks_vector[54];
	ofVec2f right_mouth_corner = face_landmarks_vector[48];

	ofVec2f mouth_center = (left_mouth_corner + right_mouth_corner) / 2;

	return mouth_center;
}

// from a set of face landmarks, get nose position. should change this to a list of ofVec2fs which have been put into screen space.
ofVec2f getNoseTip(std::vector<ofVec2f> face_landmarks_vector) {
	ofVec2f nose_pt = face_landmarks_vector[30];
	return nose_pt;
}

// left eye center in screen space 
ofVec2f getEyeLeftCenter(std::vector<ofVec2f> face_landmarks_vector) {
	ofVec2f left_eye_right_pt = face_landmarks_vector[42];
	ofVec2f left_eye_left_pt = face_landmarks_vector[45];

	ofVec2f left_eye_center = (left_eye_right_pt + left_eye_left_pt) / 2;

	return left_eye_center;
}

// right eye center in screen space 
ofVec2f getEyeRightCenter(std::vector<ofVec2f> face_landmarks_vector) {
	ofVec2f right_eye_right_pt = face_landmarks_vector[36];
	ofVec2f right_eye_left_pt = face_landmarks_vector[39];

	ofVec2f right_eye_center = (right_eye_right_pt + right_eye_left_pt) / 2;

	return right_eye_center;
}

// sets up the points in the platonic head model for head pose estimation 
// we try to convert into screen space (unclear how successfully so far!)
//std::vector<cv::Point3d> initialize_head_model_points() {
//	std::vector<cv::Point3d> model_points_temp;
//
//	// we are dividing by the original width and height of the example image in dlib,
//	// which was used for calculating these model points 
//	// we're also (currently) using the x scale to rescale the z numbers 
//	// we might also attempt to flip the y coordinates to right side up in oF space 
//	float x_scale = get_cam_width() / 1200.0f;
//	float y_scale = get_cam_height() / 675.0f;
//	float z_scale = y_scale;
//
//	model_points_temp.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
//	model_points_temp.push_back(cv::Point3d(0.0f * x_scale, -330.0f * y_scale, -65.0f * z_scale));          // Chin
//	model_points_temp.push_back(cv::Point3d(-225.0f * x_scale, 170.0f * y_scale, -135.0f * z_scale));       // Left eye left corner
//	model_points_temp.push_back(cv::Point3d(225.0f * x_scale, 170.0f * y_scale, -135.0f * z_scale));        // Right eye right corner
//	model_points_temp.push_back(cv::Point3d(-150.0f * x_scale, -150.0f * y_scale, -125.0f * z_scale));      // Left Mouth corner
//	model_points_temp.push_back(cv::Point3d(150.0f * x_scale, -150.0f * y_scale, -125.0f * z_scale));       // Right mouth corner
//	return model_points_temp;
//}

// Calculates rotation matrix to euler angles
//ofVec3f rotationMatrixToEulerAngles(cv::Mat &R)
//{
//
//	//assert(isRotationMatrix(R));
//
//	float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
//
//	bool singular = sy < 1e-6; // If
//
//	float x, y, z;
//	if (!singular)
//	{
//		x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
//		y = atan2(-R.at<double>(2, 0), sy);
//		z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
//	}
//	else
//	{
//		x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
//		y = atan2(-R.at<double>(2, 0), sy);
//		z = 0;
//	}
//	return ofVec3f(x, y, z);
//}

// calculates rotation matrix to quaternion 
//ofQuaternion rotationMatrixToQuaternion(cv::Mat &R) {
//	ofQuaternion quat;
//	ofVec4f q;
//	float trace = R.at<float>(0,0) + R.at<float>(1, 1) + R.at<float>(2, 2); 
//		if (trace > 0) {// I changed M_EPSILON to 0
//			float s = 0.5f / sqrtf(trace + 1.0f);
//			q.w = 0.25f / s;
//			q.x = (R.at<float>(2, 1) - R.at<float>(1, 2)) * s;
//			q.y = (R.at<float>(0, 2) - R.at<float>(2, 0)) * s;
//			q.z = (R.at<float>(1, 0) - R.at<float>(0, 1)) * s;
//		}
//		else {
//			if (R.at<float>(0, 0) > R.at<float>(1, 1) && R.at<float>(0, 0) > R.at<float>(2, 2)) {
//				float s = 2.0f * sqrtf(1.0f + R.at<float>(0, 0) - R.at<float>(1, 1) - R.at<float>(2, 2));
//				q.w = (R.at<float>(2, 1) - R.at<float>(1, 2)) / s;
//				q.x = 0.25f * s;
//				q.y = (R.at<float>(0, 1) + R.at<float>(1, 0)) / s;
//				q.z = (R.at<float>(0, 2) + R.at<float>(2, 0)) / s;
//			}
//			else if (R.at<float>(1, 1) > R.at<float>(2, 2)) {
//				float s = 2.0f * sqrtf(1.0f + R.at<float>(1, 1) - R.at<float>(0, 0) - R.at<float>(2, 2));
//				q.w = (R.at<float>(0, 2) - R.at<float>(2, 0)) / s;
//				q.x = (R.at<float>(0, 1) + R.at<float>(1, 0)) / s;
//				q.y = 0.25f * s;
//				q.z = (R.at<float>(1, 2) + R.at<float>(2, 1)) / s;
//			}
//			else {
//				float s = 2.0f * sqrtf(1.0f + R.at<float>(2, 2) - R.at<float>(0, 0) - R.at<float>(1, 1));
//				q.w = (R.at<float>(1, 0) - R.at<float>(0, 1)) / s;
//				q.x = (R.at<float>(0, 2) + R.at<float>(2, 0)) / s;
//				q.y = (R.at<float>(1, 2) + R.at<float>(2, 1)) / s;
//				q.z = 0.25f * s;
//			}
//		}
//		quat.set(q);
//	return quat;
//}

// rotation matrix to quaternion, opencv version
//ofQuaternion rotationMatrixToQuaternion2(cv::Mat R)
//{
//	ofVec4f q;
//	double trace = R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2);
//
//	if (trace > 0.0)
//	{
//		double s = sqrt(trace + 1.0);
//		q[3] = (s * 0.5);
//		s = 0.5 / s;
//		q[0] = ((R.at<double>(2, 1) - R.at<double>(1, 2)) * s);
//		q[1] = ((R.at<double>(0, 2) - R.at<double>(2, 0)) * s);
//		q[2] = ((R.at<double>(1, 0) - R.at<double>(0, 1)) * s);
//	}
//
//	else
//	{
//		int i = R.at<double>(0, 0) < R.at<double>(1, 1) ? (R.at<double>(1, 1) < R.at<double>(2, 2) ? 2 : 1) : (R.at<double>(0, 0) < R.at<double>(2, 2) ? 2 : 0);
//		int j = (i + 1) % 3;
//		int k = (i + 2) % 3;
//
//		double s = sqrt(R.at<double>(i, i) - R.at<double>(j, j) - R.at<double>(k, k) + 1.0);
//		q[i] = s * 0.5;
//		s = 0.5 / s;
//
//		q[3] = (R.at<double>(k, j) - R.at<double>(j, k)) * s;
//		q[j] = (R.at<double>(j, i) + R.at<double>(i, j)) * s;
//		q[k] = (R.at<double>(k, i) + R.at<double>(i, k)) * s;
//	}
//	ofQuaternion quat;
//	quat.set(q);
//
//	return quat;
//}

// Solve for head pose
//ofVec3f get_head_rotation(std::vector<cv::Point3d> model_points_i, std::vector<cv::Point2d> head_orientation_points_i) {
//	// Camera internals
//	double focal_length = get_cam_width(); // Approximate focal length. 
//	// we'll probably want to change focal_length to be the actual width of the camera input analyzed, not necessarily the width of the output texture
//
//	Point2d camera_center = cv::Point2d(get_cam_width() / 2, get_cam_height() / 2);
//	cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, camera_center.x, 0, focal_length, camera_center.y, 0, 0, 1);
//	cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion
//
//	// Output rotation and translation
//	cv::Mat rotation_vector; // Rotation in axis-angle form
//	cv::solvePnP(model_points_i, head_orientation_points_i, camera_matrix, dist_coeffs, rotation_vector, translation_vector);
//
//	//cout << translation_vector.at<float>(0, 0) << "\t" << translation_vector.at<float>(0, 1) << "\t" << translation_vector.at<float>(0, 2) << endl;
//	//cout << rotation_vector.at<float>(0, 1) << "\t" << rotation_vector.at<float>(1, 1) << "\t" << rotation_vector.at<float>(2, 1) << endl;
//	//cout << translation_vector.at<float>(0, 1) << "\t" << translation_vector.at<float>(1, 1) << "\t" << translation_vector.at<float>(2, 1) << endl;
//	//cout << "rotation vector rows: " << rotation_vector.rows << " and cols: " << rotation_vector.cols << endl;
//	//cout << "rotation value: " << rotation_vector.at<double>(0, 0);
//	//cout << "rotation vector: " << rotation_vector << endl;
//	//cout << "translation vector: " << translation_vector << endl;
//
//	// project point to 2d;
//	if (nose_end_point3D.size() < 1) {
//		nose_end_point3D.push_back(Point3d(0, 0, 1000.0));
//		nose_end_point3D.push_back(Point3d(-300, 300, 300));
//		nose_end_point3D.push_back(Point3d(300, 300, 300));
//		nose_end_point3D.push_back(Point3d(300, -300, 300));
//		nose_end_point3D.push_back(Point3d(-300, -300, 300));
//	}
//	projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);
//
//	// obtain euler angles using method found in https://answers.opencv.org/question/16796/computing-attituderoll-pitch-yaw-from-solvepnp/?answer=52913#post-id-52913
//	// obtain rotation matrix
//	cv::Mat rotation_matrix;
//	Rodrigues(rotation_vector, rotation_matrix);
//
//	cv::Vec3d eulerAngles;
//
//	cv::Mat cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ;
//	double* _r = rotation_matrix.ptr<double>();
//	double projMatrix[12] = { _r[0],_r[1],_r[2],0,
//						  _r[3],_r[4],_r[5],0,
//						  _r[6],_r[7],_r[8],0 };
//
//	decomposeProjectionMatrix(Mat(3, 4, CV_64FC1, projMatrix),
//		cameraMatrix,
//		rotMatrix,
//		transVect,
//		rotMatrixX,
//		rotMatrixY,
//		rotMatrixZ,
//		eulerAngles);
//
//	//cout << eulerAngles[0] << " " << eulerAngles[1] << " " << eulerAngles[2] << endl;
//
//	// other possible method
//	//cv::Mat projection_matrix;
//	//// append translation_vector to rotation_matrix
//	//hconcat(rotation_matrix, translation_vector, rotation_matrix);
//	//// obtain projection matrix
//	//projection_matrix = camera_matrix.dot(rotation_matrix);
//	//// obtain euler angles
//	//cv::Mat euler_angles;
//	//decomposeProjectionMatrix(projection_matrix, euler_angles)
//	//angles = cv2.decomposeProjectionMatrix(projection_matrix)[-1];
//
//
//
//	//return rotationMatrixToEulerAngles(rotation_vector);
//	//return ofVec3f(rotation_vector.at<double>(0, 0), rotation_vector.at<double>(1, 0), rotation_vector.at<double>(2, 0));
//	return ofVec3f(eulerAngles[0], eulerAngles[1], eulerAngles[2]);
//}

// converts ofPixels to a dlib compatible image format 
dlib::array2d<dlib::rgb_pixel> toDLib(const ofPixels px)
{
	dlib::array2d<dlib::rgb_pixel> out;
	int width = px.getWidth();
	int height = px.getHeight();
	int ch = px.getNumChannels();

	out.set_size(height, width);
	const unsigned char* data = px.getData();
	for (unsigned n = 0; n < height; n++)
	{
		const unsigned char* v = &data[n * width *  ch];
		for (unsigned m = 0; m < width; m++)
		{
			if (ch == 1)
			{
				unsigned char p = v[m];
				dlib::assign_pixel(out[n][m], p);
			}
			else {
				dlib::rgb_pixel p;
				p.red = v[m * 3];
				p.green = v[m * 3 + 1];
				p.blue = v[m * 3 + 2];
				dlib::assign_pixel(out[n][m], p);
			}
		}
	}
	return out;
}


//--------------------------------------------------------------
void ofApp::setup(){

	// ~~~~~~~~~
	// set up general oF business
	// ~~~~~~~~~
	ofSetFrameRate(30);
	// seems to be necessary for textures but whom knows 
	// NEVERMIND this is wildly unhelpful
	// ofDisableArbTex();
	// ofEnableArbTex();

	// ~~~~~~~~~~
	// set up camera and output FBO and difference FBOs
	// ~~~~~~~~~~
	assign_camera_IDs();
	open_cam(starting_cam);
	//allocate_difference_fbos();
	
	// ~~~~~~~~~~
	// set up spout receiver from max 
	// ~~~~~~~~~~
	//spout_receiver_max1.setup();

	// ~~~~~~~~~~~
	// set up osc receiver from max
	// ~~~~~~~~~~~

	osc_receiver.setup(osc_receiver_port);
	osc_sender.setup("127.0.0.1", osc_sender_port);

	// ~~~~~~~~~
	// frame queue
	// ~~~~~~~~~
	frame_queue.setup(get_cam_width(), get_cam_height(), texture_queue_size);

	// ~~~~~~~~~~~~~~~~~~~
	// set up face mesh 
	// ~~~~~~~~~~~~~~~~~~~
	// first we open file with texture coordinates 
	std::vector<string> face_landmark_texcoord_strings;
	ofBuffer buffer = ofBufferFromFile("face_landmark_coords.txt");
	for (auto line : buffer.getLines()) {
		face_landmark_texcoord_strings.push_back(line);
	}
	for (int i = 0; i < face_landmark_texcoord_strings.size(); i++) {
		string texcoord_string = face_landmark_texcoord_strings[i];
		std::vector<string> texcoord_temp = ofSplitString(texcoord_string, " ");
		if (texcoord_temp.size() == 2) {
			ofVec2f texcoord_vec = ofVec2f(stoi(texcoord_temp[0]), stoi(texcoord_temp[1]));
			face_texcoords.push_back(texcoord_vec);
			face_mesh.addVertex(ofPoint(0, 0)); // temporary point - this will be updated later 
			face_mesh.addTexCoord(texcoord_vec);

			// update the face texture mesh 
			face_texture_mesh.addVertex(ofPoint(texcoord_vec));
			face_texture_mesh.addTexCoord(ofVec2f(0, 0)); // temporary point, gets filled in when face detected 
		}
		else {
			cout << "malformed texture coordinate at line " << i << endl;
		}
	}
	// then we add the indices to properly connect the face points 
	std::vector<string> face_landmark_triangle_strings;
	ofBuffer face_triangle_buffer = ofBufferFromFile("face_triangulated_indices.txt");
	for (auto line : face_triangle_buffer.getLines()) {
		face_landmark_triangle_strings.push_back(line);
	}
	for (int i = 0; i < face_landmark_triangle_strings.size(); i++) {
		string triangle_string = face_landmark_triangle_strings[i];
		std::vector<string> triangle_indices_temp = ofSplitString(triangle_string, " ");
		if (triangle_indices_temp.size() == 3) {
			int tri_index_1 = stoi(triangle_indices_temp[0]);
			int tri_index_2 = stoi(triangle_indices_temp[1]);
			int tri_index_3 = stoi(triangle_indices_temp[2]);

			face_mesh.addIndex(tri_index_1);
			face_mesh.addIndex(tri_index_2);
			face_mesh.addIndex(tri_index_3);

			face_texture_mesh.addIndex(tri_index_1);
			face_texture_mesh.addIndex(tri_index_2);
			face_texture_mesh.addIndex(tri_index_3);

			//cout << "line " << i << " indices " << tri_index_1 << " " << tri_index_2 << " " << tri_index_3 << endl;
		}
		else {
			cout << "malformed triangle indices at line " << i << endl;
		}
	}
	// set proper drawing mode for face_mesh
	face_mesh.setMode(OF_PRIMITIVE_TRIANGLES);
	face_texture_mesh.setMode(OF_PRIMITIVE_TRIANGLES);

	// ~~~~~~~~~~
	// set up shaders
	// ~~~~~~~~~~

	load_shaders();
	cout << "loaded shaders" << endl;

	// ~~~~~~~~~~~~~~~
	// set up face model for head orientation 
	// ~~~~~~~~~~~~~~~
	//head_model_points = initialize_head_model_points();

	// ~~~~~~~~~~~~~~
	// set up face masks and textures 
	// ~~~~~~~~~~~~~~

	maskEyeRight.loadImage("maskEyeRight1.png");
	maskEyeLeft.loadImage("maskEyeLeft1.png");
	maskMouth.loadImage("maskMouth1.png");
	maskNose.loadImage("maskNose1.png");
	maskFace.loadImage("maskFace1.png");
	face_texture_fbo.allocate(maskEyeRight.getWidth(), maskEyeRight.getHeight(), GL_RGBA, 2);

	// read in landmark file for identifying face points 
	// loads as a command line argument
	deserialize("C:/Code/dlib-19.17/examples/build/Release/shape_predictor_68_face_landmarks.dat") >> sp;

	cv_color_img.allocate(320, 240);
	cv_grey_img.allocate(320, 240);
	cv_src_pixels.allocate(320, 240, OF_IMAGE_COLOR);

	cout << "got through setup" << endl;
}

//--------------------------------------------------------------
void ofApp::update(){

	// ~~~~~~
	// update camera and crop it into new fbo
	// ~~~~~~
	cam_grabber.update();

	cout << "updated camera" << endl;

	//ofFbo fbo_t;

	//cam_grabber.getPixels().crop(cam_bounds.x, cam_bounds.y, get_cam_width(), get_cam_height());

	cout << "cropped pixels" << endl;

	// ~~~~~~~~~
	// resize camera for analysis 
	// ~~~~~~~~~
	ofPixels cam_image_resized = cam_grabber.getPixels();

	float addl_scalar = face_detection_crop_zoom;
	cam_image_resized.crop((get_cam_width() - get_cam_width()/ addl_scalar)/2, (get_cam_height() - get_cam_height() / addl_scalar) / 2, get_cam_width() / addl_scalar, get_cam_height() / addl_scalar);

	// resize image and detect face in dlib
	cam_image_resized.resize(get_cam_width()*face_detection_scalar, get_cam_height()*face_detection_scalar);

	// convert resized image to something dlib compatible 
	camera_img_dlib_compatible = toDLib(cam_image_resized);

	cout << "made image dlib compatible" << endl;

	// ~~~~~~~~~~
	// detect face locations
	// ~~~~~~~~~~
	if (true) {
		detected_faces = face_detector(camera_img_dlib_compatible);
	}
	
	cout << "attempted face detection" << endl;

	// ~~~~~~~~~
	// update max texture
	// seems to need to be in update() or it won't work? don't put it in draw()!
	// ~~~~~~~~~
	//spout_receiver_max1.updateTexture();

	// ~~~~~~~~
	// grab the necessary params from max osc 
	// ~~~~~~~~

	if (osc_receiver.isListening()) {
		while (osc_receiver.hasWaitingMessages()) {
			ofxOscMessage incoming_message;
			osc_receiver.getNextMessage(incoming_message);
			
			// ~~~~~~~~
			// change camera
			// ~~~~~~~~

			if (incoming_message.getAddress() == "/camera_source") {
				int camera = incoming_message.getArgAsInt(0);
				open_cam(camera);
			}

			// ~~~~~~~~~~~~
			// change record mode 
			// ~~~~~~~~~~~~

			if (incoming_message.getAddress() == "/record_mode") {
				record_mode = incoming_message.getArgAsString(0);
			}

			// ~~~~~~~~
			// change mouth shape
			// ~~~~~~~~

			if (incoming_message.getAddress() == "/mouth_open_amt") {
				mouth_open_amt = incoming_message.getArgAsFloat(0);
			}
		}
	}

	cout << "attempted to receive oSC messages" << endl;

	//mouth_open_amount = osc_receiver.getNextMessage();
}

//--------------------------------------------------------------
void ofApp::draw(){
	
	cout << "starting draw" << endl;

	has_face = false;

	// check if frame passes difference test
	is_frame_different = checkIfFrameDifferent();

	cout << "checked if frame different" << endl;

	FaceDetectionFrame current_queue_frame;

	// clear the overlay fbo before drawing 
	overlay_fbo.begin();
	ofClear(0);
	overlay_fbo.end();

	mask_fbo.begin();
	ofClear(0);
	ofBackground(0, 0, 0);
	mask_fbo.end();

	face_landmarks.clear();

	// ~~~~~~~~~~
	// if the frame is different, check if we have a face
	// ~~~~~~~~~~~

	if (is_frame_different) {
		cout << "frame is different, checking for face" << endl;

		// get current face, if it exists. set to landmarks.
		// if it doesn't exist, update booleans appropriately.
		for (unsigned long j = 0; j < detected_faces.size(); ++j)
		{
			// shape: contains the landmarks for a given face
			full_object_detection shape = sp(camera_img_dlib_compatible, detected_faces[j]);

			if (shape.num_parts() == 68) {
				has_face = true;
			}

			// ~~~~~~~~~~~~
			// iterate over the shape vectors, adding them to the face_landmarks vector 
			// also adding them to face_mesh as vertices 
			// and also adding them to triangulation as points
			// vertices are converted into screen space before being added 
			// ~~~~~~~~~~~~

			for (int k = 0; k < shape.num_parts(); k++) {
				float xpos = shape.part(k).x() * 1. / (face_detection_scalar * face_detection_crop_zoom);
				float ypos = shape.part(k).y() * 1. / (face_detection_scalar * face_detection_crop_zoom);

				// add to face landmarks list
				ofVec2f current_point = ofVec2f(xpos, ypos);
				face_landmarks.push_back(current_point);
			}
		}
	}
	// ~~~~~~~~~~~~~~~
	// add frame to the queue IF it passes all criteria 
	// ~~~~~~~~~~~~~~~
	if (shouldFrameBeRecorded()) {
		cout << "frame should be recorded" << endl;
		// storing the frame - which shouldn't necessarily happen. check if we have a face first.
		frame_queue.storeFrame(cam_grabber.getTexture());
		current_queue_frame = frame_queue.getCurrentFrame();

		/*drawing_input = true;*/
		// add frame to queue 
		total_queued_textures += 1;
		last_stored_tex_queue_index = (last_stored_tex_queue_index + 1) % min(total_queued_textures, texture_queue_size);
		ofFbo storage_fbo = cam_tex_queue[last_stored_tex_queue_index];
		storage_fbo.begin();
		//ofBackground(0);
		//cam_grabber.getTexture().draw(0, 0);
		current_queue_frame.getTexture().draw(0, 0);
		storage_fbo.end();

		// set texture to current grabber
		selected_texture = current_queue_frame.getTexture();

		// store landmarks in queue /IF/ we are adding the frame to the queue 
		if (has_face) {
			frame_queue.addFaceToCurrentFrame(face_landmarks);
		}
	}
	else {
		cout << "frame shouldn't be recorded - time to grab one from queue" << endl;
		// we have 
		current_queue_frame = frame_queue.getNextFrame();
		selected_texture = current_queue_frame.getTexture();

		if (previous_tex_queue_index >= 0) {
			ofFbo previous_stored_tex = cam_tex_queue[previous_tex_queue_index];

			// draw difference between current and looped textures 
			difference_fbo.begin();
			shader_absdiff.begin();
			shader_absdiff.setUniformTexture("tex0", current_queue_frame.getFbo().getTextureReference(), 0);
			shader_absdiff.setUniformTexture("tex1", previous_stored_tex.getTextureReference(), 1);
			selected_texture.draw(0, 0);
			shader_absdiff.end();
			difference_fbo.end();
			difference_fbo.draw(get_cam_width(), 0);

			// check if it has landmarks, and if so, grab them 
		}
		if (current_queue_frame.hasFace()) {
			has_face = true;
			face_landmarks = current_queue_frame.getFaceLandmarks();
		}
		
		current_queue_frame.printLandmarks();
	}

	// read to opencv contour finder
	/*selected_texture.readToPixels(cv_src_pixels);
	cv_src_pixels.resize(320, 240);
	cv_src_pixels.setImageType(OF_IMAGE_COLOR);
	cv_color_img.setFromPixels(cv_src_pixels);
	cv_grey_img.setFromColorImage(cv_color_img);
	cv_grey_img.threshold(150);
	cv_contour_finder.findContours(cv_grey_img, 30, 10000, 20, false, true);*/

	output_fbo.begin();

	cout << "starting to draw" << endl;

	// ~~~~~~~~~
	// draw background and camera image 
	// ~~~~~~~~~

	ofBackground(0);
	//cam_grabber.getTexture().drawSubsection(0, 0, get_cam_width(), get_cam_height(), cam_bounds.x, cam_bounds.y, cam_bounds.z, cam_bounds.w);
	// draw selected tex

	selected_texture.draw(0, 0);

	ofDrawBitmapString(ofGetFrameRate(), 30, 30);

	// Now we will go ask the shape_predictor to tell us the pose of
	// the faces we've detected 

	if(has_face) {

		cout << "we have a face, let's draw it" << endl;
		// ~~~~~~~~~~~~~
		// update face texture with detected face
		// ~~~~~~~~~~~~~
		updateFaceMeshVertices(face_landmarks);

		//ofFbo cam_fbo;
		//cam_fbo.allocate(get_cam_width(), get_cam_height(), GL_RGBA, 1);
		//cam_fbo.begin();
		//cam_grabber.getTexture().drawSubsection(0, 0, get_cam_width(), get_cam_height(), cam_bounds.x, cam_bounds.y, cam_bounds.z, cam_bounds.w);
		//cam_fbo.end();

		face_texture_fbo.begin();
		ofClear(0);
		cam_grabber.getTexture().bind();
		face_texture_mesh.draw();
		cam_grabber.getTexture().unbind();
		face_texture_fbo.end();

		// test eye centers 
		ofVec2f eye_right_center = getEyeRightCenter(face_landmarks);
		ofVec2f eye_left_center = getEyeLeftCenter(face_landmarks);
		ofVec2f mouth_center = getMouthCenter(face_landmarks);
		ofVec2f nose_center = getNoseTip(face_landmarks);

		// drawing eye stalks 
		for (int i = 0; i < 20; i++) {
			float x_offset = sin((ofGetFrameNum() + i) / 10.) * i * 7;
			float x_offset2 = sin((ofGetFrameNum() + i) / 8.3) * i * 7;
			float scale_factor = abs(1. - (i / 50.));
			//drawBodyPart(maskEyeLeft, scale_factor, eye_left_center, eye_left_center+ofVec2f(x_offset,0));
			//drawBodyPart(maskEyeRight, scale_factor, eye_right_center, eye_right_center+ofVec2f(x_offset2, 0));
			//drawBodyPart(maskMouth, scale_factor, mouth_center, mouth_center + ofVec2f(0, x_offset));
		}

		// ~~~~~~~~
		// draw face mask
		// ~~~~~~~~

		// we use color to (roughly) identify the nose center
		ofPushStyle();
		float mask_red = 255 * mouth_center.x / get_cam_width();
		float mask_green = 255 * mouth_center.y / get_cam_height();
		ofSetColor(mask_red, mask_green, 255);
		// draw to mask fbo 
		mask_fbo.begin();
		ofBackground(mask_red, mask_green, 0);
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		maskFace.getTexture().bind();
		face_mesh.draw();
		maskFace.getTexture().unbind();
		ofEnableAlphaBlending();
		mask_fbo.end();
		ofPopStyle();

		// draw mask on openframeworks window (not necessary)
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		maskFace.getTexture().bind();
		face_mesh.draw();
		maskFace.getTexture().unbind();
		ofEnableAlphaBlending();

		// ~~~~~~~~~~~~
		// test out the bubble shader
		// ~~~~~~~~~~~~
		if (false) {
			shader_eye_distortion.begin();
			shader_eye_distortion.setUniform2f("bubble_center", getNoseTip(face_landmarks).x / get_cam_width(), getNoseTip(face_landmarks).y / get_cam_height());
			cam_grabber.draw(0, 0, get_cam_width(), get_cam_height());
			shader_eye_distortion.end();
		}

		// ~~~~~~~~~~~~~~~~~~
		// draw the face mesh with the bound spout texture 
		// ~~~~~~~~~~~~~~~~~~

		if (false) {
			//spout_receiver_max1.getTexture().bind();
			//face_mesh.draw();
			//spout_receiver_max1.getTexture().unbind();
		}


		// ~~~~~~~~~~~~~~
		// record the alignment of the face
		// used when making masks 
		// we record the image with wireframe overlaid
		// also the texture coordinates as a list
		// and the indices of the face triangles as a list 
		// ~~~~~~~~~~~~~~~

		if (record_face_alignment) {
			// find bounds of face landmarks
			std::vector<ofVec2f> face_landmarks_cropped;
			float smallest_landmark_x = get_cam_width() + 1;
			float largest_landmark_x = -1;
			float smallest_landmark_y = get_cam_height() + 1;
			float largest_landmark_y = -1;
			// honestly we could do this part while assigning face_landmarks above, if we're not going to mirror 
			for (int i = 0; i < face_landmarks.size(); i++) {
				ofVec2f current_pt = face_landmarks[i];
				float current_x = current_pt.x;
				float current_y = current_pt.y;

				smallest_landmark_x = min(smallest_landmark_x, current_x);
				largest_landmark_x = max(largest_landmark_x, current_x);
				smallest_landmark_y = min(smallest_landmark_y, current_y);
				largest_landmark_y = max(largest_landmark_y, current_y);
			}

			// do the triangulation 
			face_triangulation.triangulate();

			// write triangulation indices to a file 
			ofFile face_triangulated_indices_file;
			face_triangulated_indices_file.open("face_triangulated_indices.txt", ofFile::WriteOnly);
			for (int i = 0; i < face_triangulation.triangleMesh.getNumIndices() / 3; i++) {
				int i_mult = i * 3;
				int tri_index_1 = face_triangulation.triangleMesh.getIndex(i_mult);
				int tri_index_2 = face_triangulation.triangleMesh.getIndex(i_mult+1);
				int tri_index_3 = face_triangulation.triangleMesh.getIndex(i_mult+2);

				face_triangulated_indices_file << tri_index_1 << " " << tri_index_2 << " " << tri_index_3 << endl;
			}

			// adjust face landmarks so they're aligned to the left and top 
			ofFile face_landmark_coords_file;
			face_landmark_coords_file.open("face_landmark_coords.txt", ofFile::WriteOnly);
			ofFbo face_landmarks_image;
			face_landmarks_image.allocate(largest_landmark_x - smallest_landmark_x, largest_landmark_y - smallest_landmark_y, GL_RGBA, 4);
			face_landmarks_image.begin();
			ofBackground(0);
			cam_grabber.draw(-smallest_landmark_x, -smallest_landmark_y, get_cam_width(), get_cam_height());
			for (int i = 0; i < face_landmarks.size(); i++) {
				ofVec2f current_pt = face_landmarks[i];
				ofVec2f bounded_pt = current_pt - ofVec2f(smallest_landmark_x, smallest_landmark_y);
				face_landmarks_cropped.push_back(bounded_pt);

				face_landmark_coords_file << bounded_pt.x << " " << bounded_pt.y << endl;

				ofEllipse(bounded_pt.x, bounded_pt.y, 3, 3);
			}

			ofPushMatrix();
			ofTranslate(-smallest_landmark_x, -smallest_landmark_y);
			face_triangulation.triangleMesh.drawWireframe();
			ofPopMatrix();

			face_landmarks_image.end();
			ofImage face_landmarks_image_file;
			ofPixels face_landmarks_pixels;
			face_landmarks_image.readToPixels(face_landmarks_pixels);
			face_landmarks_image_file.setFromPixels(face_landmarks_pixels);
			face_landmarks_image_file.save("face_landmarks_image.png");
			record_face_alignment = false;
		}

		// ~~~~~~~~~~~
		// attempt to obtain the face alignment and draw it
		// ~~~~~~~~~~~

		if (false) {
			//ofVec2f nose_pos = getNoseTip(shape);
			//ofPushMatrix();
			//model_points = init_model_points();
			//head_orientation_points = extract_head_orientation_points(shape);
			//ofQuaternion nose_quat = solve_for_head_rotation(model_points, head_orientation_points);
			//float nose_angle;
			//ofVec3f nose_angle_vec;
			//nose_quat.getRotate(nose_angle, nose_angle_vec);
			//ofRotate(nose_angle, nose_angle_vec.x, nose_angle_vec.y, nose_angle_vec.z);
			////ofLine(ofVec3f(nose_pos.x, nose_pos.y, 0), ofVec3f( nose_pos.x, nose_pos.y, 5));
			//ofPopMatrix();

			//ofPushStyle();
			//ofSetColor(255, 0, 0);
			////ofDrawCircle(nose_pos.x*4., nose_pos.y*4., 3.);
			//ofDrawLine(nose_pos.x*4., nose_pos.y*4., nose_end_point2D[0].x * 4., nose_end_point2D[1].y * 4.);
			//ofPopStyle();
		}

		// ~~~~~~~~~~~~~~~~~
		// make meshes and draw eyes 
		// ~~~~~~~~~~~~~~~~~

		if (false) {
			//std::vector<ofVec2f> eyeCoords;
			//std::vector<ofVec2f> eyeTexCoords;
			//face_triangulation.reset();

			//// if we have enough parts, find and draw the center of the right eye
			//if (shape.num_parts() == 68) {
			//	int num_verts = 6;
			//	int vert_offset = 36;
			//	ofVec2f avg_position = ofVec2f(0, 0);
			//	for (int i = 0; i < num_verts; i++) {
			//		int index = i + vert_offset;
			//		float xpos = shape.part(index).x() * 2.;
			//		float ypos = shape.part(index).y() * 2.;
			//		ofVec2f pos = ofVec2f(xpos, ypos);
			//		avg_position += pos;

			//		eyeCoords.push_back(pos);
			//		eyeTexCoords.push_back(ofVec2f(pos.x / 1280., pos.y / 720.));
			//	}
			//	avg_position /= num_verts;

			//	// make eyes bigger
			//	for (int i = 0; i < num_verts; i++) {
			//		int index = i + vert_offset;
			//		float xpos = shape.part(index).x() * 2.;
			//		float ypos = shape.part(index).y() * 2.;
			//		ofVec2f pos = ofVec2f(xpos, ypos);

			//		ofVec2f diff = pos - avg_position;
			//		ofVec2f newPos = diff * 2 + avg_position;

			//		eyeCoords[i] = newPos;

			//		ofVec2f avg_pos_normal = avg_position / ofVec2f(get_cam_width(), get_cam_height());
			//		ofVec2f newTex = (eyeTexCoords[i] - avg_pos_normal) * 2 + avg_pos_normal;
			//		eyeTexCoords[i] = newTex;
			//	}

			//	ofVec2f eye_corner_1 = ofVec2f(shape.part(36).x(), shape.part(36).y());
			//	ofVec2f eye_corner_2 = ofVec2f(shape.part(39).x(), shape.part(39).y());

			//	float eye_distance = eye_corner_1.distance(eye_corner_2) * 2.;

			//	/*		ofSetColor(255);
			//			ofEllipse(avg_position.x, avg_position.y, eye_distance*2., eye_distance*2.);
			//			ofSetColor(0);
			//			ofEllipse(avg_position.x, avg_position.y, eye_distance*1., eye_distance*1.);
			//			ofSetColor(255);*/

			//}
			//// left eye
			//if (shape.num_parts() == 68) {
			//	int num_verts = 6;
			//	int vert_offset = 42;
			//	ofVec2f avg_position = ofVec2f(0, 0);
			//	for (int i = 0; i < num_verts; i++) {
			//		int index = i + vert_offset;
			//		float xpos = shape.part(index).x() * 2.;
			//		float ypos = shape.part(index).y() * 2.;
			//		ofVec2f pos = ofVec2f(xpos, ypos);
			//		avg_position += pos;

			//		eyeCoords.push_back(pos);
			//		eyeTexCoords.push_back(ofVec2f(pos.x / get_cam_width(), pos.y / get_cam_height()));
			//	}
			//	avg_position /= num_verts;

			//	// make eyes bigger
			//	for (int i = 0; i < num_verts; i++) {
			//		int index = i + vert_offset;
			//		float xpos = shape.part(index).x() * 2.;
			//		float ypos = shape.part(index).y() * 2.;
			//		ofVec2f pos = ofVec2f(xpos, ypos);

			//		ofVec2f diff = pos - avg_position;
			//		ofVec2f newPos = diff * 2 + avg_position;

			//		ofVec2f avg_pos_normal = avg_position / ofVec2f(1280, 720);
			//		ofVec2f newTex = (eyeTexCoords[i + 6] - avg_pos_normal) * 2 + avg_pos_normal;
			//		eyeTexCoords[i + 6] = newTex;

			//		eyeCoords[i + 6] = newPos;
			//	}

			//	ofVec2f eye_corner_1 = ofVec2f(shape.part(36).x(), shape.part(36).y());
			//	ofVec2f eye_corner_2 = ofVec2f(shape.part(39).x(), shape.part(39).y());

			//	float eye_distance = eye_corner_1.distance(eye_corner_2) * 2.;

			//	/*ofSetColor(255);
			//	ofEllipse(avg_position.x, avg_position.y, eye_distance*2., eye_distance*2.);
			//	ofSetColor(0);
			//	ofEllipse(avg_position.x, avg_position.y, eye_distance*1., eye_distance*1.);
			//	ofSetColor(255);*/
			//}

			//// weird eyes
			////for (int rec = 0; rec < 13; rec++) {
			////	for (int i = 0; i < eyeCoords.size(); i++) {
			////		ofVec2f eyePos = eyeCoords[i];
			////		if (rec > 0) eyePos += ofVec2f(cos(rec / 6. * PI * 2. + ofGetFrameNum() / 20.)*150, sin(rec / 6. * PI * 2. + +ofGetFrameNum() / 20.)*150);
			////		if (rec > 6) eyePos += ofVec2f(cos(rec / 6. * PI * 2.) * 150, sin(rec / 6. * PI * 2.) * 150);
			////		triangulation.addPoint(ofPoint(eyePos));
			////	}
			////}

			////triangulation.addPoint(ofPoint(0, 0));
			////triangulation.addPoint(ofPoint(ofGetWidth(), 0));
			////triangulation.addPoint(ofPoint(0, ofGetHeight()));
			////triangulation.addPoint(ofPoint(ofGetWidth(), ofGetHeight()));

			////triangulation.triangulate();

			////for (int rec = 0; rec < 13; rec++) {
			////	for (int i = 0; i < eyeCoords.size(); i++) {
			////		ofVec2f eyeTex = eyeTexCoords[i];
			////		triangulation.triangleMesh.addTexCoord(eyeTex);
			////	}
			////}

			////triangulation.triangleMesh.addTexCoord(ofVec2f(0, 0));
			////triangulation.triangleMesh.addTexCoord(ofVec2f(1, 0));
			////triangulation.triangleMesh.addTexCoord(ofVec2f(0, 1));
			////triangulation.triangleMesh.addTexCoord(ofVec2f(1, 1));

			////ofTexture grabTexx;
			////grabTexx.loadData(grabber.getPixels());

			////grabTexx.bind();
			////triangulation.triangleMesh.draw();
			////grabTexx.unbind();
		}

		// ~~~~~~~~
		// draw lips
		// ~~~~~~~~

		if (true) {
			// 54 and 48 are the mouth corners
			// 60 and 64 the inner corners 

			lips.clear();
			mouth.clear();

			// get mouth angle 
			ofVec2f right_mouth_centered = face_landmarks[54] - mouth_center;
			ofVec2f horizontal_axis = ofVec2f(1, 0);
			float mouth_angle = horizontal_axis.angle(right_mouth_centered);
			float mouth_scale = face_landmarks[54].distance(face_landmarks[48]);
			ofVec2f mouth_offset = ofVec2f(0, 1);
			mouth_offset.rotate(mouth_angle);
			//float mouth_open_scale = sin(ofGetFrameNum() / 10) * 0.5 + 0.5;
			float mouth_open_scale = mouth_open_amt;
			float mouth_open_upper = mouth_open_scale * -mouth_scale * 0.3;
			float mouth_open_lower = mouth_open_scale * mouth_scale * 0.3;

			// add lips
			for (int i = 48; i < 68; i++) {
				ofVec2f pt = face_landmarks[i];

				// attempt to scale mouth by moving top lip 
				if ((i >= 49 && i <= 53) || (i >= 61 && i <= 63)) {
					pt -= mouth_center;
					pt *= mouth_open_scale * 0 + 1;
					pt += mouth_center;
					pt += mouth_offset * mouth_open_upper;
				}
				if ((i >= 55 && i <= 59) || (i >= 65 && i <= 67)) {
					pt -= mouth_center;
					pt *= mouth_open_scale * 0 + 1;
					pt += mouth_center;
					pt += mouth_offset * mouth_open_lower;
				}

				lips.addVertex(ofPoint(pt));

				// add to mouth 
				if (i >= 60 && i < 68) {
					ofVec2f tc = face_landmarks[i];
					mouth.addVertex(ofPoint(pt));
					//ofVec2f texc = ofVec2f(tc.x / get_cam_width(), tc.y / get_cam_height());
					//ofVec2f texc = tc;
					//mouth.addTexCoord(texc);
				}
			}
			lips.addTriangle(0, 1, 12);
			lips.addTriangle(1, 12, 13);
			lips.addTriangle(1, 2, 13);
			lips.addTriangle(2, 3, 13);
			lips.addTriangle(3, 13, 14);
			lips.addTriangle(3, 14, 15);
			lips.addTriangle(3, 4, 15);
			lips.addTriangle(4, 5, 15);
			lips.addTriangle(5, 15, 16);
			lips.addTriangle(5, 6, 16);
			lips.addTriangle(6, 7, 16);
			lips.addTriangle(7, 16, 17);
			lips.addTriangle(7, 8, 17);
			lips.addTriangle(8, 17, 18);
			lips.addTriangle(8, 9, 18);
			lips.addTriangle(9, 10, 18);
			lips.addTriangle(10, 18, 19);
			lips.addTriangle(10, 11, 19);
			lips.addTriangle(11, 12, 19);
			lips.addTriangle(0, 11, 12);

			// add mouth triangulation
			mouth.addTriangle(0, 1, 7);
			mouth.addTriangle(1, 2, 7);
			mouth.addTriangle(2, 6, 7);
			mouth.addTriangle(2, 3, 6);
			mouth.addTriangle(3, 5, 6);
			mouth.addTriangle(3, 4, 5);

			overlay_fbo.begin();

			// draw lips
			ofSetColor(255, 0, 0);
			lips.draw();
			ofSetColor(255);

			// use the color to specify mouth position / mask for (yet more) mask feedback 
			ofPushStyle();
			ofSetColor(0);
			mouth.draw();
			ofPopStyle();

			overlay_fbo.end();

			overlay_fbo.draw(0, 0);
		}

		// test drawing the model face
		if (false) {
			//ofPushMatrix();
			//ofTranslate(get_cam_width() / 2, get_cam_height() / 2);
			//ofRotateY(ofGetFrameNum());
			//for (int i = 0; i < head_model_points.size(); i++) {
			//	float xp = head_model_points[i].x;
			//	float yp = head_model_points[i].y;
			//	float zp = head_model_points[i].z;
			//	ofPushStyle();
			//	// draw eyes as green, all else as red
			//	if (i == 2 || i == 3) {
			//		ofSetColor(0, 255, 0);
			//	}
			//	else {
			//		ofSetColor(255, 0, 0);
			//	}
			//	//ofDrawSphere(xp, yp, zp, 5);
			//	ofDrawLine(0, 0, 0, 0, 0, 1000);
			//	ofPopStyle();
			//}
			//ofPopMatrix();
			//ofDrawBitmapString(ofGetFrameNum()%360, 50, 200);

		}

		// try to draw the head rotation
		if (false) {
			//ofVec3f head_rotation = get_head_rotation(head_model_points, get_head_orientation_points(face_landmarks));
			////ofVec3f head_rotation = head_rotation_q.getEuler();
			//ofVec2f nose_position = getNoseTip(face_landmarks);
			//ofPushMatrix();
			////ofTranslate(nose_position.x, nose_position.y);
			//ofTranslate(get_cam_width()/2, get_cam_height()/2);
			//ofTranslate(-translation_vector.at<double>(0, 0) * 2.75, -translation_vector.at<double>(1, 0) * 2.75, translation_vector.at<double>(2, 0) * 1);
			//ofRotateX(-head_rotation.x);
			//ofRotateY(-head_rotation.y);
			//ofRotateZ(head_rotation.z);
			////ofRotateX(ofRadToDeg(head_rotation.x));
			////ofRotateY(ofRadToDeg(-head_rotation.z));
			////ofRotateZ(ofRadToDeg(head_rotation.z));
			////cout << "head rotation y " << head_rotation.y << endl;
			//ofPushStyle();
			//ofSetColor(255, 0, 0);
			//
			///*ofTranslate(0,0,250);
			//ofDrawBox(0, 0, 0, 50, 50, 500);*/

			//// display w/ recursive boxes
			//ofMesh boxMesh;
			//boxMesh.addVertex(ofPoint(-500,800));
			//boxMesh.addVertex(ofPoint(500,800));
			//boxMesh.addVertex(ofPoint(500,-800));
			//boxMesh.addVertex(ofPoint(-500,-800));
			//boxMesh.setMode(OF_PRIMITIVE_LINE_LOOP);
			//for (int c = 0; c < 20; c++) {
			//	ofTranslate(0, 0, 50);
			//	//if ((ofGetFrameNum() / 4 + c) % 4 == 0) boxMesh.draw();
			//	boxMesh.draw();
			//}

			////ofDrawLine(nose_position.x, nose_position.y, nose_end_point2D[0].x, nose_end_point2D[0].y);
			////ofDrawLine(nose_end_point2D[0].x, nose_end_point2D[0].y, nose_end_point2D[1].x, nose_end_point2D[1].y);
			////ofDrawLine(nose_end_point2D[0].x, nose_end_point2D[0].y, nose_end_point2D[3].x, nose_end_point2D[3].y);
			////ofDrawLine(nose_end_point2D[2].x, nose_end_point2D[2].y, nose_end_point2D[1].x, nose_end_point2D[1].y);
			////ofDrawLine(nose_end_point2D[2].x, nose_end_point2D[2].y, nose_end_point2D[3].x, nose_end_point2D[3].y);
			//ofPopStyle();
			//ofPopMatrix();

			////ofDrawLine(nose_position.x, nose_position.y, nose_end_point2D[0].x, nose_end_point2D[0].y);
		}
		
	}
	output_fbo.end();
	output_fbo.draw(0, 0);

	//cv_contour_finder.draw(0, 0, get_cam_width(), get_cam_height());
	//for (int i = 0; i < cv_contour_finder.blobs.size(); i++) {
	//	ofxCvBlob blob = cv_contour_finder.blobs[i];
	//	float x_pos = blob.centroid.x;
	//	float y_pos = blob.centroid.y;
	//	float area = blob.area;
	//	ofxOscMessage centroid_message;
	//	centroid_message.setAddress("/blob_centroids");
	//	centroid_message.addIntArg(i);
	//	centroid_message.addFloatArg(x_pos / 320.);
	//	centroid_message.addFloatArg(y_pos / 240.);
	//	centroid_message.addFloatArg(area / (320. * 240.));
	//	osc_sender.sendMessage(centroid_message);
	//}

	// ~~~~~~~~~~~
	// send all spout textures 
	// ~~~~~~~~~~~

	cout << "finished drawing, time to send to spout" << endl;

	//spout_sender.sendTexture(selected_texture, "image_texture");
	//spout_sender.sendTexture(difference_fbo.getTexture(), "difference_map");
	//spout_sender.sendTexture(mask_fbo.getTexture(), "face_mask");
	//spout_sender.sendTexture(overlay_fbo.getTexture(), "overlay");

	cout << "sent textures to spout" << endl;
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	//if (key == 'r') {
	//	seq_recorder.startRecording();
	//}
	if (key == 'l') {
		current_highlighted_landmark += 1;
		current_highlighted_landmark %= 68;
	}
	if (key == 'k') {
		current_highlighted_landmark += 68 - 1;
		current_highlighted_landmark %= 68;
	}
	if (key == 'f') {
		//record_face_alignment = true;
	}
	if (key == 's') {
		load_shaders();
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    