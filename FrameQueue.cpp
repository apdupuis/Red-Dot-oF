#include "FrameQueue.h"



FrameQueue::FrameQueue()
{
}


FrameQueue::~FrameQueue()
{
}

void FrameQueue::setup(int fbo_width, int fbo_height, int queue_size)
{
	max_queue_size = queue_size;
	total_stored_frames = 0;
	last_retrieved_index = -1;
	last_stored_index = -1;

	frames_queue.clear();
	for (int i = 0; i < queue_size; i++) {
		FaceDetectionFrame new_frame;
		new_frame.setup(fbo_width, fbo_height);
		frames_queue.push_back(new_frame);
	}
}

void FrameQueue::storeFrame(ofTexture new_tex)
{
	total_stored_frames += 1;
	int store_index = (last_stored_index + 1) % min(max_queue_size, total_stored_frames);
	FaceDetectionFrame face_frame = frames_queue[store_index];
	face_frame.storeTexture(new_tex);
	frames_queue[store_index] = face_frame;
	last_stored_index = store_index;

	//cout << "adding texture to frame " << store_index << endl;
}

void FrameQueue::addFaceToCurrentFrame(std::vector<ofVec2f> new_landmarks)
{
	FaceDetectionFrame current_frame = getCurrentFrame();
	current_frame.storeLandmarks(new_landmarks);
	setCurrentFrame(current_frame);

	//cout << "adding a face to frame " << last_stored_index << endl;
}

FaceDetectionFrame FrameQueue::getNextFrame()
{
	last_retrieved_index = (last_retrieved_index + 1) % min(max_queue_size, total_stored_frames);
	return frames_queue[last_retrieved_index];
}

FaceDetectionFrame FrameQueue::getCurrentFrame()
{
	return frames_queue[last_stored_index];
}

void FrameQueue::setCurrentFrame(FaceDetectionFrame new_frame)
{
	frames_queue[last_stored_index] = new_frame;
}

int FrameQueue::getLastStoredIndex()
{
	return last_stored_index;
}
