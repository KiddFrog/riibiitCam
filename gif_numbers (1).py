from PIL import Image, ImageSequence

def rearrange_and_move_frames(input_gif_path, output_gif_path):
    # Open the GIF image
    gif = Image.open(input_gif_path)
    frames = []

    try:
        while True:
            frames.append(gif.copy().convert('RGBA'))
            gif.seek(len(frames))  # Go to the next frame
    except EOFError:
        pass

    # Rearrange the order of frames
    new_order = [0, 1, 3, 2, 3, 1]
    rearranged_frames = [frames[i] for i in new_order]

    # Create a new image with the specified dimensions
    new_gif = Image.new('RGBA', (2328, 1748), (255, 255, 255, 0))

    for i, frame in enumerate(rearranged_frames):
        # Move frames based on the specified instructions
        if i == 0:  # FRAME 1
            frame = frame.crop((102, 66, 2430, 1814))  # Move right 102, down 66
        elif i == 2:  # FRAME 3
            frame = frame.crop((154, 24, 2482, 1772))  # Move up 24, left 154
        elif i == 3:  # FRAME 4
            frame = frame.crop((244, 34, 2572, 1782))  # Move down 34, left 244

        # Paste the frame onto the new image
        new_gif.paste(frame, (0, 0), frame)

    # Save the new GIF
    new_gif.save(output_gif_path, save_all=True, append_images=frames[1:], loop=0)

if __name__ == "__main__":
    input_gif_path = "/home/froggo/Desktop/PICTURES/test.gif"
    output_gif_path = "/home/froggo/Desktop/PICTURES/output_test.gif"
    rearrange_and_move_frames(input_gif_path, output_gif_path)


    ## As far as the cameras are aligned now the order is
## RX0 LEFT LEFT (edge)
## RX1 LEFT
## RX2 RIGHT
## RX3 RIGHT RIGHT (edge)
## However, when imported as a gif, the order is 1 > 2 > 4 > 3 (Wonder if I should switch the position of RX2 && RX3)


## I've found that in my experiments RX1 (Camera 2, one from the left) is a great baseline

## DIMENSIONS from PHOTOSHOP
## 2328px by 1748px  (odd dimensions)

## So lets figure out 
## FRAME 1 -- MOVE RIGHT 102 px MOVE DOWN 66 px
## FRAME 2 -- UNCHANGED ** BASELINE **
## FRAME 3 -- MOVE UP 24 px MOVE LEFT 154 px ** THIS SHOULD BE SWAPPED WITH FRAME 4!!! **
## FRAME 4 -- MOVE DOWN 34 px MOVE LEFT 244 px ** THIS SHOULD BE SWAPPED WITH FRAME 3!!! **
## PLAY FRAME 3, PLAY FRAME 2, loop
