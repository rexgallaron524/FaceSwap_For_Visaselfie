# Reference capture and enrollment

FaceLive starts with eight required slots:

| Slot | Capture direction |
| --- | --- |
| Front neutral | Face the camera with a relaxed expression |
| Front smile | Face the camera with a natural smile |
| Left 20° / Left 40° | Turn your head approximately 20° / 40° to your left |
| Right 20° / Right 40° | Turn your head approximately 20° / 40° to your right |
| Up / Down | Face forward and look slightly up / down |

Left and right always mean the subject's direction in an unmirrored image. A phone's
mirrored selfie preview can be misleading; use the pose requested by the slot label.

## Recommended conditions

- Use recent images of the same consenting person for every slot.
- Capture at 1280×720 or higher when possible. The face should be at least 96 pixels wide
  and high, occupy a useful part of the frame, and remain fully visible.
- Use soft, even front lighting. Avoid a bright window behind the subject, hard shadows,
  clipped highlights, and heavy color filters.
- Keep camera height, distance, focal length, background, hairstyle, and accessories
  consistent across the set.
- Keep both eyes visible. Remove sunglasses and avoid hands, hair, masks, or objects that
  cover the eyes, nose, mouth, jaw, or cheeks.
- Use a sharp still image. Motion blur, digital zoom, beauty filters, and screenshots of
  compressed video reduce landmark and texture quality.
- Include exactly one face. Crop out bystanders before enrollment.

The automated checks enforce file readability, image dimensions, exactly one detected
face, a complete landmark mesh, minimum face size and coverage, usable exposure, and basic
sharpness. Pose labels and subject consistency remain user responsibilities in this
milestone.

## Enrollment workflow

1. Start FaceLive and select **Manage references…**.
2. Choose an image for each card. A valid image immediately shows its normalized thumbnail.
3. Replace or remove any image as needed. Rejected images display a specific reason.
4. Select **Save library…**. FaceLive writes a JSON manifest and a neighboring
   `<name>_assets` directory containing the normalized PNG files.
5. Use **Load library…** later to restore the validated cache without running face detection
   again.

Source images are read during enrollment and are not copied into the saved library. The
library contains normalized face imagery, landmark geometry, blendshape values, and source
file hashes. Treat the manifest and asset directory as sensitive biometric data, keep both
together, and store or share them only with the subject's consent.

## Validation and preprocessing

MediaPipe Face Landmarker runs locally in still-image mode. The adapter returns a 478-point
mesh and 52 blendshape values. FaceLive uses eye landmarks to remove in-plane rotation and
scale the face into an immutable 512×512 RGB reference. It records normalized landmarks,
source dimensions and digest, face bounds and coverage, brightness, sharpness, alignment
rotation, and creation time.

All work happens when the image is enrolled. Saving stores that prepared result, and loading
verifies the asset checksum and compatibility before exposing it to future selection and
rendering stages. No reference detection or alignment runs in the live camera loop.
