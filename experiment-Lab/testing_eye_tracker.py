import pylink
from psychopy import visual, core, event

# Connect to EyeLink tracker
tracker = pylink.EyeLink()

# Set up PsychoPy window
win = visual.Window([1920, 1080], fullscr=True, color=(0, 0, 0), units="pix")

# Create a visual marker to show gaze position (e.g., a small circle)
gaze_marker = visual.Circle(win, radius=10, fillColor="red", lineColor=None)

# Start recording eye data
tracker.startRecording(1, 1, 1, 1)
core.wait(0.1)  # Small delay to ensure recording starts

# Main loop to continuously update gaze position
while not event.getKeys():  # Break the loop if a key is pressed
    # Fetch the latest eye-tracking sample
    sample = tracker.getNewestSample()

    if sample is not None:
        if sample.isRightSample():  # Assuming we are tracking the right eye
            gaze_pos = sample.getRightEye().getGaze()  # Get gaze coordinates

            # Update gaze marker position on the PsychoPy window
            gaze_marker.pos = (
                gaze_pos[0] - 960,
                540 - gaze_pos[1],
            )  # Convert to PsychoPy coordinates
            gaze_marker.draw()

    # Flip the PsychoPy window to update the display
    win.flip()

# Stop recording after the loop ends
tracker.stopRecording()

# Close the EyeLink connection and PsychoPy window
tracker.close()
win.close()
core.quit()
