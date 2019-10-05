#pragma once

#include "FaceDetectionFrame.h"

class FrameQueue
{
public:
	FrameQueue();
	~FrameQueue();

	void setup(int fbo_width, int fbo_height, int queue_size);
	void storeFrame(ofTexture new_tex);
	void addFaceToCurrentFrame(std::vector<ofVec2f> new_landmarks);
	FaceDetectionFrame getNextFrame();
	FaceDetectionFrame getCurrentFrame();

private: 
	std::vector<FaceDetectionFrame> frames_queue;
	int max_queue_size;
	int total_stored_frames; // so we don't accidentally pick a blank, frame 
	int last_stored_index;
	int last_retrieved_index;
};

