                                                                                                                                                                                                   #include "ofApp.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

#include "ofxDelaunay.h"

#include "opencv2/opencv.hpp"

using namespace dlib;
using namespace std;

using namespace cv;

// 2D image points taken from a texture, used to calculate head pose 
std::vector<cv::Point2d> head_orientation_points;

cv::Point2d extract_head_orientation_point(full_object_detection face_landmarks_shape, int index) {
	float shape_x = face_landmarks_shape.part(index).x();
	float shape_y = face_landmarks_shape.part(index).y();
	return cv::Point2d(shape_x, shape_y);
}

std::vector<cv::Point2d> extract_head_orientation_points(full_object_detection face_landmarks_shape) {
	std::vector<cv::Point2d> head_orientation_points_temp;
	head_orientation_points_temp.push_back(extract_head_orientation_point(face_landmarks_shape, 30));	// Nose tip
	head_orientation_points_temp.push_back(extract_head_orientation_point(face_landmarks_shape, 8));	// Chin
	head_orientation_points_temp.push_back(extract_head_orientation_point(face_landmarks_shape, 45));	// Left eye left corner 
	head_orientation_points_temp.push_back(extract_head_orientation_point(face_landmarks_shape, 36));	// Right eye right corner
	head_orientation_points_temp.push_back(extract_head_orientation_point(face_landmarks_shape, 54));	// Left Mouth corner
	head_orientation_points_temp.push_back(extract_head_orientation_point(face_landmarks_shape, 48));	// Right mouth corner
	return head_orientation_points_temp;
}

// the points in the arbitrary face model, used for estimating head orientation
std::vector<cv::Point3d> model_points;

std::vector<cv::Point3d> init_model_points() {
	std::vector<cv::Point3d> model_points_temp;

	float x_scale = ofGetWidth() / 1200.0f / 4.0f;
	float y_scale = ofGetHeight() / 675.0f / 4.0f;

	model_points_temp.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
	model_points_temp.push_back(cv::Point3d(0.0f * x_scale, -330.0f * y_scale, -65.0f * x_scale));          // Chin
	model_points_temp.push_back(cv::Point3d(-225.0f * x_scale, 170.0f * y_scale, -135.0f * x_scale));       // Left eye left corner
	model_points_temp.push_back(cv::Point3d(225.0f * x_scale, 170.0f * y_scale, -135.0f * x_scale));        // Right eye right corner
	model_points_temp.push_back(cv::Point3d(-150.0f * x_scale, -150.0f * y_scale, -125.0f * x_scale));      // Left Mouth corner
	model_points_temp.push_back(cv::Point3d(150.0f * x_scale, -150.0f * y_scale, -125.0f * x_scale));       // Right mouth corner
	return model_points_temp;
}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
ofVec3f rotationMatrixToEulerAngles(cv::Mat &R)
{

	assert(isRotationMatrix(R));

	float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

	bool singular = sy < 1e-6; // If

	float x, y, z;
	if (!singular)
	{
		x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
		y = atan2(-R.at<double>(2, 0), sy);
		z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
	}
	else
	{
		x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
		y = atan2(-R.at<double>(2, 0), sy);
		z = 0;
	}
	return ofVec3f(x, y, z);
}

ofQuaternion rotationMatrixToQuaternion(cv::Mat &R) {
	ofQuaternion quat;
	ofVec4f q;
	float trace = R.at<float>(0,0) + R.at<float>(1, 1) + R.at<float>(2, 2); 
		if (trace > 0) {// I changed M_EPSILON to 0
			float s = 0.5f / sqrtf(trace + 1.0f);
			q.w = 0.25f / s;
			q.x = (R.at<float>(2, 1) - R.at<float>(1, 2)) * s;
			q.y = (R.at<float>(0, 2) - R.at<float>(2, 0)) * s;
			q.z = (R.at<float>(1, 0) - R.at<float>(0, 1)) * s;
		}
		else {
			if (R.at<float>(0, 0) > R.at<float>(1, 1) && R.at<float>(0, 0) > R.at<float>(2, 2)) {
				float s = 2.0f * sqrtf(1.0f + R.at<float>(0, 0) - R.at<float>(1, 1) - R.at<float>(2, 2));
				q.w = (R.at<float>(2, 1) - R.at<float>(1, 2)) / s;
				q.x = 0.25f * s;
				q.y = (R.at<float>(0, 1) + R.at<float>(1, 0)) / s;
				q.z = (R.at<float>(0, 2) + R.at<float>(2, 0)) / s;
			}
			else if (R.at<float>(1, 1) > R.at<float>(2, 2)) {
				float s = 2.0f * sqrtf(1.0f + R.at<float>(1, 1) - R.at<float>(0, 0) - R.at<float>(2, 2));
				q.w = (R.at<float>(0, 2) - R.at<float>(2, 0)) / s;
				q.x = (R.at<float>(0, 1) + R.at<float>(1, 0)) / s;
				q.y = 0.25f * s;
				q.z = (R.at<float>(1, 2) + R.at<float>(2, 1)) / s;
			}
			else {
				float s = 2.0f * sqrtf(1.0f + R.at<float>(2, 2) - R.at<float>(0, 0) - R.at<float>(1, 1));
				q.w = (R.at<float>(1, 0) - R.at<float>(0, 1)) / s;
				q.x = (R.at<float>(0, 2) + R.at<float>(2, 0)) / s;
				q.y = (R.at<float>(1, 2) + R.at<float>(2, 1)) / s;
				q.z = 0.25f * s;
			}
		}
		quat.set(q);
	return quat;
}

std::vector<cv::Point3d> nose_end_point3D;
std::vector<cv::Point2d> nose_end_point2D;

// Solve for pose
ofQuaternion solve_for_head_rotation(std::vector<cv::Point3d> model_points_i, std::vector<cv::Point2d> head_orientation_points_i) {
	// Camera internals
	double focal_length = ofGetWidth() / 4; // Approximate focal length. 
	// we'll probably want to change focal_length to be the actual width of the camera input analyzed, not necessarily the width of the output texture

	Point2d camera_center = cv::Point2d(ofGetWidth() / (2*4), ofGetHeight() / (2*4));
	cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, camera_center.x, 0, focal_length, camera_center.y, 0, 0, 1);
	cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

	// Output rotation and translation
	cv::Mat rotation_vector; // Rotation in axis-angle form
	cv::Mat translation_vector;
	cv::solvePnP(model_points_i, head_orientation_points_i, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

	// project point to 2d;
	if (nose_end_point3D.size() < 1) {
		nose_end_point3D.push_back(Point3d(0, 0, 100.0));
	}
	projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);

	return rotationMatrixToQuaternion(rotation_vector);
}


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

// We need a face detector.  We will use this to get bounding boxes for
		// each face in an image.
frontal_face_detector detector = get_frontal_face_detector();
// And we also need a shape_predictor.  This is the tool that will predict face

array2d<rgb_pixel> img;
ofImage faces_img;
ofVideoGrabber grabber;
std::vector<dlib::rectangle> dets; // contains detections
shape_predictor sp;
ofxDelaunay triangulation;

ofTexture replace_face;
std::vector<ofVec2f> replace_face_tex_coords;

ofFbo output_fbo;
ofImage mask1;

bool record_face_alignment = false;

// the index of the currently highlighted landmark, used for determining the various landmark indices 
int current_highlighted_landmark = 0;

ofVec2f getNoseTip(full_object_detection face_landmarks_shape) {
	cv::Point2d nose_pt_cv = extract_head_orientation_point(face_landmarks_shape, 30);
	return ofVec2f(nose_pt_cv.x, nose_pt_cv.y);
}

// mesh which can be used to write over the face
std::vector<ofVec2f> face_texcoords;
ofMesh face_mesh;

//--------------------------------------------------------------
void ofApp::setup(){

	output_fbo.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA, 4);

	seq_recorder.setLength(300);
	seq_recorder.setImage(&output_fbo);

	// set up face mesh 
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

			//cout << "line " << i << " indices " << tri_index_1 << " " << tri_index_2 << " " << tri_index_3 << endl;
 		}
		else {
			cout << "malformed triangle indices at line " << i << endl;
		}
	}
	// set proper drawing mode for face_mesh
	face_mesh.setMode(OF_PRIMITIVE_TRIANGLES);

	// set up face replacement texture
	mask1.load("face_mask1.png");

	// landmark positions given an image and face bounding box.  Here we are just
// loading the model from the shape_predictor_68_face_landmarks.dat file you gave
// as a command line argument.
	deserialize("C:/Code/dlib-19.17/examples/build/Release/shape_predictor_68_face_landmarks.dat") >> sp;

	grabber.setDeviceID(1);
	grabber.setup(640, 360);

	//faces_img.load("C:/Code/dlib-19.17/examples/build/Release/faces/emotions.png");

	ofSetFrameRate(30);

	ofDisableArbTex();

	ofImage new_face;
	new_face.load("C:/Users/alexa/OneDrive/Pictures/Mine/Headshots/DaniHeadshots/Headshot1.jpg");
	replace_face.loadData(new_face.getPixels());
	img = toDLib(new_face.getPixels());
	dets = detector(img);

	full_object_detection shape = sp(img, dets[0]);
	//cout << "number of parts: " << shape.num_parts() << endl;
	//cout << "pixel position of first part:  " << shape.part(0) << endl;
	//cout << "pixel position of second part: " << shape.part(1) << endl;
	// You get the idea, you can get all the face part locations if
	// you want them.  Here we just store them in shapes so we can
	// put them on the screen.
	//shapes.push_back(shape);

	//triangulation.reset();

	//for (int k = 0; k < shape.num_parts(); k++) {
	//	float xpos = shape.part(k).x();
	//	float ypos = shape.part(k).y();
	//	ofPoint pt = ofPoint(xpos, ypos);



	//	triangulation.addPoint(pt);
	//	ofVec2f replace_tex_coord = ofVec2f(xpos / new_face.getWidth(), ypos / new_face.getHeight());
	//	replace_face_tex_coords.push_back(replace_tex_coord);

	//	//ofSetColor(0, 255, 0);
	//	//ofEllipse(xpos, ypos, 3, 3);
	//	//ofSetColor(255);
	//}
	//triangulation.triangulate();

	// set up face model for head orientation 
	model_points = init_model_points();
}

//--------------------------------------------------------------
void ofApp::update(){
	if (seq_recorder.isRecording) {
		seq_recorder.update();
	}

	grabber.update();

	ofPixels pix_tmp = grabber.getPixels();
	//pix_tmp.resize(pix_tmp.getWidth() * 0.5, pix_tmp.getHeight() * 0.5);

	pix_tmp.resize(pix_tmp.getWidth()*0.5, pix_tmp.getHeight()*0.5);

	img = toDLib(pix_tmp);

	// detect face locations
	if (ofGetFrameNum() % 1 == 0) {
		dets = detector(img);
	}
	// adjust face detection bounds if desired 
	if (dets.size() > 0) {
		//dets[0].set_left(0);
		//dets[0].set_right(pix_tmp.getWidth());
		//dets[0].set_bottom(pix_tmp.getHeight());
		//dets[0].set_top(0);
	}

}

//--------------------------------------------------------------
void ofApp::draw(){
	//if (ofGetFrameNum() < 100) {
	//	ofBackground(0);
	//}
	//faces_img.draw(0, 0);
	//grabber.draw(0, 0, ofGetWidth(), ofGetHeight());
	//ofDrawBitmapString(ofGetFrameRate(), 30, 30);

	output_fbo.begin();

	if (true) {
		ofBackground(0);
	}
	//grabber.draw(0, 0, ofGetWidth(), ofGetHeight());
	ofDrawBitmapString(ofGetFrameRate(), 30, 30);

	// Now we will go ask the shape_predictor to tell us the pose of
			// each face we detected.
	//std::vector<full_object_detection> shapes;
	for (unsigned long j = 0; j < dets.size(); ++j)
	{
		full_object_detection shape = sp(img, dets[j]);
		//cout << "number of parts: " << shape.num_parts() << endl;
		//cout << "pixel position of first part:  " << shape.part(0) << endl;
		//cout << "pixel position of second part: " << shape.part(1) << endl;
		// You get the idea, you can get all the face part locations if
		// you want them.  Here we just store them in shapes so we can
		// put them on the screen.
		//shapes.push_back(shape);

		//triangulation.reset();

		std::vector<ofVec2f> face_landmarks;

		triangulation.reset();
		for (int k = 0; k < shape.num_parts(); k++) {
			float xpos = shape.part(k).x() * 2. / 0.5;
			float ypos = shape.part(k).y() * 2. / 0.5;

			ofVec2f current_point = ofVec2f(xpos, ypos);
			face_landmarks.push_back(current_point);
			//ofPoint pt = ofPoint(xpos, ypos);

			//triangulation.triangleMesh.setVertex(k, ofPoint(current_point));
			triangulation.addPoint(ofPoint(current_point));

			face_mesh.setVertex(k, ofPoint(current_point));

			if (k == current_highlighted_landmark) {
				ofSetColor(255, 0, 0);
			}
			else {
				ofSetColor(0, 255, 0);
			}
			//ofEllipse(xpos, ypos, 3, 3);
			ofSetColor(255);
		}

		//face_mesh.drawWireframe();
		//ofEnableBlendMode(OF_BLENDMODE_SUBTRACT);
		mask1.getTexture().bind();
		face_mesh.draw();
		mask1.getTexture().unbind();
		//ofDisableBlendMode();

		// center coordinates around nose (not yet implemented 
		ofVec2f nose_coord = getNoseTip(shape);

		// find bounds of face landmarks
		std::vector<ofVec2f> face_landmarks_cropped;
		float smallest_landmark_x = ofGetWidth() + 1;
		float largest_landmark_x = -1;
		float smallest_landmark_y = ofGetHeight() + 1;
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

		if (record_face_alignment) {
			// do the triangulation 
			triangulation.triangulate();

			// check if the triangle indices and the face mesh indices align 
			//for (int i = 0; i < face_mesh.getNumVertices(); i++) {
			//	ofPoint face_mesh_vertex = face_mesh.getVertex(i);
			//	ofPoint triangulation_vertex = triangulation.triangleMesh.getVertex(i);
			//	cout << "face vertex " << i << ": " << face_mesh_vertex.x << ", " << face_mesh_vertex.y << " // tri vertex " << triangulation_vertex.x << ", " << triangulation_vertex.y << endl;
			//}

			// write triangulation indices to a file 
			ofFile face_triangulated_indices_file;
			face_triangulated_indices_file.open("face_triangulated_indices.txt", ofFile::WriteOnly);
			//for (int i = 0; i < triangulation.getNumTriangles(); i++) {
			//	ITRIANGLE tri = triangulation.getTriangleAtIndex(i);
			//	int tri_index_1 = tri.p1;
			//	int tri_index_2 = tri.p2;
			//	int tri_index_3 = tri.p3;

			//	face_triangulated_indices_file << tri_index_1 << " " << tri_index_2 << " " << tri_index_3 << endl;
			//}
			for (int i = 0; i < triangulation.triangleMesh.getNumIndices() / 3; i++) {
				int i_mult = i * 3;
				int tri_index_1 = triangulation.triangleMesh.getIndex(i_mult);
				int tri_index_2 = triangulation.triangleMesh.getIndex(i_mult+1);
				int tri_index_3 = triangulation.triangleMesh.getIndex(i_mult+2);

				face_triangulated_indices_file << tri_index_1 << " " << tri_index_2 << " " << tri_index_3 << endl;
			}

			// adjust face landmarks so they're aligned to the left and top 
			ofFile face_landmark_coords_file;
			face_landmark_coords_file.open("face_landmark_coords.txt", ofFile::WriteOnly);
			ofFbo face_landmarks_image;
			face_landmarks_image.allocate(largest_landmark_x - smallest_landmark_x, largest_landmark_y - smallest_landmark_y, GL_RGBA, 4);
			face_landmarks_image.begin();
			ofBackground(0);
			grabber.draw(-smallest_landmark_x, -smallest_landmark_y, ofGetWidth(), ofGetHeight());
			for (int i = 0; i < face_landmarks.size(); i++) {
				ofVec2f current_pt = face_landmarks[i];
				ofVec2f bounded_pt = current_pt - ofVec2f(smallest_landmark_x, smallest_landmark_y);
				face_landmarks_cropped.push_back(bounded_pt);

				face_landmark_coords_file << bounded_pt.x << " " << bounded_pt.y << endl;

				ofEllipse(bounded_pt.x, bounded_pt.y, 3, 3);
			}

			ofPushMatrix();
			ofTranslate(-smallest_landmark_x, -smallest_landmark_y);
			triangulation.triangleMesh.drawWireframe();
			ofPopMatrix();

			face_landmarks_image.end();
			ofImage face_landmarks_image_file;
			ofPixels face_landmarks_pixels;
			face_landmarks_image.readToPixels(face_landmarks_pixels);
			face_landmarks_image_file.setFromPixels(face_landmarks_pixels);
			face_landmarks_image_file.save("face_landmarks_image.png");
			record_face_alignment = false;
		}

		



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

		std::vector<ofVec2f> eyeCoords;
		std::vector<ofVec2f> eyeTexCoords;
		triangulation.reset();

		// if we have enough parts, find and draw the center of the right eye
		if (shape.num_parts() == 68) {
			int num_verts = 6;
			int vert_offset = 36;
			ofVec2f avg_position = ofVec2f(0, 0);
			for (int i = 0; i < num_verts; i++) {
				int index = i + vert_offset;
				float xpos = shape.part(index).x() * 2.;
				float ypos = shape.part(index).y() * 2.;
				ofVec2f pos = ofVec2f(xpos, ypos);
				avg_position += pos;

				eyeCoords.push_back(pos);
				eyeTexCoords.push_back(ofVec2f(pos.x / 1280., pos.y / 720.));
			}
			avg_position /= num_verts;

			// make eyes bigger
			for (int i = 0; i < num_verts; i++) {
				int index = i + vert_offset;
				float xpos = shape.part(index).x() * 2.;
				float ypos = shape.part(index).y() * 2.;
				ofVec2f pos = ofVec2f(xpos, ypos);
				
				ofVec2f diff = pos - avg_position;
				ofVec2f newPos = diff * 2 + avg_position;

				eyeCoords[i] = newPos;

				ofVec2f avg_pos_normal = avg_position / ofVec2f(1280, 720);
				ofVec2f newTex = (eyeTexCoords[i] - avg_pos_normal) * 2 + avg_pos_normal;
				eyeTexCoords[i] = newTex;
			}

			ofVec2f eye_corner_1 = ofVec2f(shape.part(36).x(), shape.part(36).y());
			ofVec2f eye_corner_2 = ofVec2f(shape.part(39).x(), shape.part(39).y());

			float eye_distance = eye_corner_1.distance(eye_corner_2) * 2.;

	/*		ofSetColor(255);
			ofEllipse(avg_position.x, avg_position.y, eye_distance*2., eye_distance*2.);
			ofSetColor(0);
			ofEllipse(avg_position.x, avg_position.y, eye_distance*1., eye_distance*1.);
			ofSetColor(255);*/

		}
		// left eye
		if (shape.num_parts() == 68) {
			int num_verts = 6;
			int vert_offset = 42;
			ofVec2f avg_position = ofVec2f(0, 0);
			for (int i = 0; i < num_verts; i++) {
				int index = i + vert_offset;
				float xpos = shape.part(index).x() * 2.;
				float ypos = shape.part(index).y() * 2.;
				ofVec2f pos = ofVec2f(xpos, ypos);
				avg_position += pos;

				eyeCoords.push_back(pos);
				eyeTexCoords.push_back(ofVec2f(pos.x / 1280., pos.y / 720.));
			}
			avg_position /= num_verts;

			// make eyes bigger
			for (int i = 0; i < num_verts; i++) {
				int index = i + vert_offset;
				float xpos = shape.part(index).x() * 2.;
				float ypos = shape.part(index).y() * 2.;
				ofVec2f pos = ofVec2f(xpos, ypos);

				ofVec2f diff = pos - avg_position;
				ofVec2f newPos = diff * 2 + avg_position;

				ofVec2f avg_pos_normal = avg_position / ofVec2f(1280, 720);
				ofVec2f newTex = (eyeTexCoords[i + 6] - avg_pos_normal) * 2 + avg_pos_normal;
				eyeTexCoords[i + 6] = newTex;

				eyeCoords[i+6] = newPos;
			}

			ofVec2f eye_corner_1 = ofVec2f(shape.part(36).x(), shape.part(36).y());
			ofVec2f eye_corner_2 = ofVec2f(shape.part(39).x(), shape.part(39).y());

			float eye_distance = eye_corner_1.distance(eye_corner_2) * 2.;

			/*ofSetColor(255);
			ofEllipse(avg_position.x, avg_position.y, eye_distance*2., eye_distance*2.);
			ofSetColor(0);
			ofEllipse(avg_position.x, avg_position.y, eye_distance*1., eye_distance*1.);
			ofSetColor(255);*/
		}

		// weird eyes
		//for (int rec = 0; rec < 13; rec++) {
		//	for (int i = 0; i < eyeCoords.size(); i++) {
		//		ofVec2f eyePos = eyeCoords[i];
		//		if (rec > 0) eyePos += ofVec2f(cos(rec / 6. * PI * 2. + ofGetFrameNum() / 20.)*150, sin(rec / 6. * PI * 2. + +ofGetFrameNum() / 20.)*150);
		//		if (rec > 6) eyePos += ofVec2f(cos(rec / 6. * PI * 2.) * 150, sin(rec / 6. * PI * 2.) * 150);
		//		triangulation.addPoint(ofPoint(eyePos));
		//	}
		//}

		//triangulation.addPoint(ofPoint(0, 0));
		//triangulation.addPoint(ofPoint(ofGetWidth(), 0));
		//triangulation.addPoint(ofPoint(0, ofGetHeight()));
		//triangulation.addPoint(ofPoint(ofGetWidth(), ofGetHeight()));

		//triangulation.triangulate();

		//for (int rec = 0; rec < 13; rec++) {
		//	for (int i = 0; i < eyeCoords.size(); i++) {
		//		ofVec2f eyeTex = eyeTexCoords[i];
		//		triangulation.triangleMesh.addTexCoord(eyeTex);
		//	}
		//}

		//triangulation.triangleMesh.addTexCoord(ofVec2f(0, 0));
		//triangulation.triangleMesh.addTexCoord(ofVec2f(1, 0));
		//triangulation.triangleMesh.addTexCoord(ofVec2f(0, 1));
		//triangulation.triangleMesh.addTexCoord(ofVec2f(1, 1));

		//ofTexture grabTexx;
		//grabTexx.loadData(grabber.getPixels());

		//grabTexx.bind();
		//triangulation.triangleMesh.draw();
		//grabTexx.unbind();

		// lips 
		if (shape.num_parts() == 68) {
			ofMesh lips;
			ofMesh mouth;

			// add lips
			for (int i = 48; i < 68; i++) {
				ofVec2f pt = ofVec2f(shape.part(i).x(), shape.part(i).y()) * 2.;
				lips.addVertex(ofPoint(pt));
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

			// draw lips
			ofSetColor(255, 0, 0);
			//lips.draw();
			ofSetColor(255);

			// add mouth
			for (int i = 60; i < 68; i++) {
				ofVec2f pt = ofVec2f(shape.part(i).x(), shape.part(i).y()) * 2.;
				mouth.addVertex(ofPoint(pt));
				ofVec2f texc = ofVec2f(pt.x / 1280., pt.y / 720.);
				mouth.addTexCoord(texc);
			}

			mouth.addTriangle(0, 1, 7);
			mouth.addTriangle(1, 2, 7);
			mouth.addTriangle(2, 6, 7);
			mouth.addTriangle(2, 3, 6);
			mouth.addTriangle(3, 5, 6);
			mouth.addTriangle(3, 4, 5);

			//cout << mouth.getTexCoord(5).x << " " << mouth.getTexCoord(5).y << endl;

			ofTexture grabTex;
			grabTex.loadData(grabber.getPixels());

			grabTex.bind();
			mouth.draw();
			grabTex.unbind();
		}

		//triangulation.triangulate();
		
		for (int i = 0; i < triangulation.triangleMesh.getNumVertices(); i++) {
			ofPoint pt = triangulation.triangleMesh.getVertex(i);
			float x_new = pt.x / 1280.;
			float y_new = pt.y / 720.;
			ofVec2f tex = ofVec2f(x_new, y_new);
			tex = replace_face_tex_coords[i];
			triangulation.triangleMesh.addTexCoord(tex);
		}



		replace_face.bind();
		//triangulation.triangleMesh.draw();
		replace_face.unbind();

		
	}
	output_fbo.end();
	output_fbo.draw(0, 0);

	ofDrawBitmapString(current_highlighted_landmark, 50, 100);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	if (key == 'r') {
		seq_recorder.startRecording();
	}
	if (key == 'l') {
		current_highlighted_landmark += 1;
		current_highlighted_landmark %= 68;
	}
	if (key == 'k') {
		current_highlighted_landmark += 68 - 1;
		current_highlighted_landmark %= 68;
	}
	if (key == 'f') {
		record_face_alignment = true;
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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    