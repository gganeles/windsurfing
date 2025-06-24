import cv2
import numpy as np
import subprocess
import sys
from datetime import datetime
import json
import ffmpeg
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, Polygon
from itertools import combinations
from scipy.stats import gaussian_kde

def perpendicular_distance_lines_to_point(rho_theta: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """
    Calculates the perpendicular distances from multiple lines (defined by rho, theta in Hough space)
    to a specific point (x0, y0).

    Parameters:
        rho_theta (np.ndarray): Nx2 array where each row is [rho, theta].
        x0 (float): x-coordinate of the point.
        y0 (float): y-coordinate of the point.

    Returns:
        np.ndarray: Array of perpendicular distances for each line.
    """
    # Extract rho and theta from the input array
    rho = rho_theta[:, 0]
    theta = rho_theta[:, 1]

    # Compute the perpendicular distances for each (rho, theta)
    distances = np.abs(x0 * np.cos(theta) + y0 * np.sin(theta) - rho)
    return distances

def find_indices_within_mode_threshold(data, threshold):
    """
    Finds the indices of values within a specified threshold of the mode.

    Parameters:
        data (np.ndarray): Input array of float values.
        threshold (float): Threshold range around the mode.

    Returns:
        np.ndarray: Indices of values within the threshold of the mode.
    """
    if len(data) == 0:
        raise ValueError("Input data array is empty.")
    
    # Calculate the mode using KDE
    kde = gaussian_kde(data)
    x = np.linspace(np.min(data), np.max(data), 1000)
    density = kde(x)
    mode_kde = x[np.argmax(density)]

    # Determine indices of values within the threshold
    lower_bound = mode_kde - threshold
    upper_bound = mode_kde + threshold
    indices_within_threshold = np.where((data >= lower_bound) & (data <= upper_bound))[0]

    return indices_within_threshold

def hough_line_transform(edges, num_rho=180, num_theta=180):
    """
    Perform Hough Line Transform to find the strongest line in the image.

    Parameters:
        edges (np.ndarray): Binary edge-detected image (2D array).
        num_rho (int): Number of bins for rho (distance).
        num_theta (int): Number of bins for theta (angle).

    Returns:
        tuple: (rho, theta) of the strongest line.
    """
    # Get the dimensions of the image
    height, width = edges.shape

    # Define rho and theta ranges
    diag_len = int(np.sqrt(height**2 + width**2))  # Max distance
    rho_range = np.linspace(-diag_len, diag_len, num_rho)
    theta_range = np.linspace(-np.pi, np.pi, num_theta)

    # Create the accumulator
    accumulator = np.zeros((len(rho_range), len(theta_range)), dtype=np.int32)

    # Find edge points
    edge_points = np.argwhere(edges)

    # Populate the accumulator
    for y, x in edge_points:
        for theta_index, theta in enumerate(theta_range):
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_index = np.argmin(np.abs(rho_range - rho))
            accumulator[rho_index, theta_index] += 1

    # Find the maximum value in the accumulator
    max_index = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    strongest_rho = rho_range[max_index[0]]
    strongest_theta = theta_range[max_index[1]]

    return strongest_rho, strongest_theta

def detect_horizon_line(image_grayscaled,num_lines,mode='combinations',xshift=0):
        """Detect the horizon's starting and ending points in the given image

        The horizon line is detected by applying Otsu's threshold method to
        separate the sky from the remainder of the image.

        :param image_grayscaled: grayscaled image to detect the horizon on, of
        shape (height, width)
        :type image_grayscale: np.ndarray of dtype uint8
        :return: the (x1, x2, y1, y2) coordinates for the starting and ending
        points of the detected horizon line
        :rtype: tuple(int)
        """

        plot = "plot" in sys.argv
        v = True
        
        msg = ('`image_grayscaled` should be a grayscale, 2-dimensional image '
            'of shape (height, width).')
        
        houghRectSize = (15, 15)
        
        assert image_grayscaled.ndim == 2, msg  
        
        
        for x in [
            (30,30)
        ]:
            if plot and v:
                plt.imshow(image_grayscaled)
                plt.show()
            
            if False:
                image_grayscaled = cv2.GaussianBlur(image_grayscaled, (21,21), 5)
                #image_blurred = cv2.blur(image_grayscaled, (15,15))
            
            _, image_thresholded = cv2.threshold(
                image_grayscaled, thresh=0, maxval=1, 
                type=cv2.THRESH_BINARY+cv2.THRESH_OTSU
                )
            
            # _, image_thresholded2 = cv2.threshold(
            #     image_grayscaled, thresh=150, maxval=255,
            #     type=cv2.THRESH_BINARY_INV
            #     )
            
            image_thresholded = image_thresholded - 1
            
            # if plot:
            #     plt.imshow(image_thresholded)
            #     plt.show()
            #     plt.imshow(image_thresholded2)
            #     plt.show()
            
            #triangle_points = np.array([
            #     [0,image_thresholded.shape[0]], 
            #     [270, 351], 
            #     [470, 346],
            #     [image_thresholded.shape[1],382],
            #     [image_thresholded.shape[1],image_thresholded.shape[0]]])

            # Fill the triangle on the mask
            #masked_image = cv2.fillPoly(image_thresholded, [triangle_points], 255)
    
            
            image_closed = cv2.morphologyEx(image_thresholded, cv2.MORPH_CLOSE,
                                    kernel=np.ones(x, np.uint8))
            

            num = 2*num_lines
            xVec = np.linspace(0,image_grayscaled.shape[1] - 1,num).astype(int)
            #yVec = yvec[xVec]

            board_mask = np.array([
                [0,image_thresholded.shape[0]], 
                [0,407],
                [270, 351], 
                [470, 346],
                [768,382],
                [768,image_thresholded.shape[0]]])
            
            board_mask[:,0] = board_mask[:,0] - xshift*np.ones(board_mask.shape[0])
            yVec = dy_of_1_above_thresh(image_closed[:,xVec],board_mask,xVec)

            
            #yVec = list(map(lambda x: highestDarkPixelWithFillCheck(image_closed, x, 20, plot),xVec))

            if plot and v:
                plt.imshow(image_closed)
                plt.scatter(xVec,yVec)
                plt.plot(board_mask[:,0],board_mask[:,1])
                plt.show()

            def filter_lines_by_theta(lines, theta):
                """
                Filters lines based on their angle with respect to the x-axis.
                
                Parameters:
                    lines (list): A list of lines, where each line is represented as [[x1, y1], [x2, y2]].
                    theta (float): The threshold angle in degrees. Lines outside the range [-theta, theta] are removed.
                    
                Returns:
                    list: A filtered list of lines within the specified angle range.
                """
                filtered_lines = []
                theta_rad = theta  # Convert theta to radians

                for line in lines:
                    (x1, y1), (x2, y2) = line
                    dx = x2 - x1
                    dy = y2 - y1

                    # Calculate the angle of the line in radians
                    angle = np.arctan(dy/dx)

                    # Check if the angle is within the range
                    if -theta_rad <= angle <= theta_rad:
                        filtered_lines.append(line)

                return np.array(filtered_lines)

            def compute_line_similarity_matrix(lines):
                """
                Compute similarity matrix between line vectors using dot products.
                
                Parameters:
                lines: List of line endpoints, where each line is [[x1,y1], [x2,y2]]
                
                Returns:
                similarity_matrix: Matrix of dot products between normalized direction vectors
                direction_vectors: The normalized direction vectors for each line
                """

                # Calculate direction vectors (end point - start point)
                direction_vectors = lines[:, 1] - lines[:, 0]  # shape: (n_lines, 2)
                #direction_vectors = lines
                # Normalize the direction vectors
                norms = np.linalg.norm(direction_vectors, axis=1, keepdims=True)
                normalized_vectors = direction_vectors / norms
                            
                # Compute dot product matrix
                # This is equivalent to normalized_vectors @ normalized_vectors.T
                similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
                
                return similarity_matrix
            
            def find_most_similar_row(matrix, threshold):
                """
                Find the row in the matrix that has the most values above the given threshold.
                
                Parameters:
                matrix: numpy array of shape (n, n)
                threshold: float value (typically close to 1) to use as cutoff
                
                Returns:
                most_similar_row_idx: Index of the row with most values above threshold
                count: Number of values above threshold in the winning row
                mask: Boolean array showing which elements in the winning row were above threshold
                """
                # Count how many values in each row are above the threshold
                above_threshold = matrix >= threshold
                
                # Sum across rows to get count for each row
                counts = np.sum(above_threshold, axis=1)
                
                # Find the row with the maximum count
                most_similar_row_idx = np.argmax(counts)
                
                # Get the mask for the winning row
                winning_row_mask = above_threshold[most_similar_row_idx]
                
                # Get the count for the winning row
                #winning_count = counts[most_similar_row_idx]
                
                return most_similar_row_idx, winning_row_mask

            if mode=="hough":
                
                yVec = yVec.astype(int)
                image = np.zeros(image_closed.shape, dtype=np.uint8)
                
                real_points = np.where(yVec!=0)
                cleaned_points = np.array((yVec,xVec)).T[real_points].T

                # Set the specified points to 1
                image[cleaned_points[0],cleaned_points[1]] = 1            

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, houghRectSize)

                horizonEdgeDialated = cv2.dilate(image,kernel)
                
                #rho, theta = hough_line_transform(horizonEdgeDialated,30,100)
                lines = cv2.HoughLines(horizonEdgeDialated, rho=1, theta=np.pi / 180, threshold=60)
                
                # plt.imshow(horizonEdgeDialated)
                # t = np.linspace(0,image_closed.shape[1],2)
                # plt.scatter(cleaned_points[1],cleaned_points[0],c="g")
                # plt.show()
                
                if not np.any(lines):
                    return False, False
                rho, theta = lines[0][0]
                
            else:
                lines = []
                
                for i in range(num//2):
                    if yVec[i] and yVec[num-i-1]:
                        lines.append(np.array([
                            [xVec[i],yVec[i]],
                            [xVec[num-i-1],yVec[num-i-1]]
                        ]))  
                               
                if mode=='combinations':
                    lines = np.array(list(combinations(np.array(list(zip(xVec,yVec))),2)))
                      
                
                #lineArray = np.array(lines)

                lineArray = filter_lines_by_theta(lines,np.pi/3)

                if len(lineArray)==0:
                    return False, False

                matrix = compute_line_similarity_matrix(lineArray)

                x, mask = find_most_similar_row(matrix, .99995)
                
                lines = lineArray[mask]
                
                rho = np.median(abs((lines[:,1,0] * lines[:,0,1] - lines[:,0,0] * lines[:,1,1])) / np.sqrt((lines[:,1,0] - lines[:,0,0])**2 + (lines[:,1,1] - lines[:,0,1])**2))
                
                theta = np.mean(np.arctan((lines[:,1,1]-lines[:,0,1])/(lines[:,1,0]-lines[:,0,0])))+np.pi/2
                
            if plot:
                if mode=="hough":
                    plt.imshow(horizonEdgeDialated)
                    t = np.linspace(0,image_closed.shape[1],2)
                    plt.plot(t,houghLineFunction(t,rho,theta))
                    plt.scatter(cleaned_points[1],cleaned_points[0],c="g")
                    plt.show()
                else:
                    plt.imshow(image_closed)
                    for line in lines:
                        plt.plot(line[:,0],line[:,1])
                    plt.plot(board_mask[:,0],board_mask[:,1])
                    plt.show()
            return rho, theta

def highestDarkPixelWithFillCheck(img,x,buffer = 10,plot=False):
    max_y = img.shape[0] - 1
    points = np.where(img[:, x] == 0)
    if len(points[0]) > 0:
        y = max(points[0])
    else:
        return False
    
    
    if y>=max_y-buffer:
        image_flooded = img.copy()
        _,image_flooded,_,_ = cv2.floodFill(image_flooded, None, (x, y), 255)
        points = np.where(image_flooded[:, x] == 0)
        
        if len(points[0]) > 0:
            y = max(points[0])
            if y >= max_y-buffer:
                return False
        else:
            return False
    return y
            
def detect_horizon_line_hough(image_grayscaled,threshold=150):
    plot = False
    v = False
    
    msg = ('`image_grayscaled` should be a grayscale, 2-dimensional image '
        'of shape (height, width).')
    
    assert image_grayscaled.ndim == 2, msg  
    
    
    for x in [
        (10,10),
    ]:
        if plot and v:
            plt.imshow(image_grayscaled)
            plt.show()
        
        if False:
            image_grayscaled = cv2.GaussianBlur(image_grayscaled, (21,21), 5)
            #image_blurred = cv2.blur(image_grayscaled, (15,15))
        
        _, image_thresholded = cv2.threshold(
            image_grayscaled, thresh=0, maxval=1, 
            type=cv2.THRESH_BINARY+cv2.THRESH_OTSU
            )
        
        _, image_thresholded2 = cv2.threshold(
            image_grayscaled, thresh=threshold, maxval=255,
            type=cv2.THRESH_BINARY_INV
            )
        
        image_thresholded = image_thresholded - 1
        
        if plot:
            plt.imshow(image_thresholded2)
            plt.show()
        
        mask =False
        if mask:
            triangle_points = np.array([
                [0,image_thresholded.shape[0]], 
                [270, 351], 
                [470, 346],
                [image_thresholded.shape[1],382],
                [image_thresholded.shape[1],image_thresholded.shape[0]]])

            # Fill the triangle on the mask
            masked_image = cv2.fillPoly(image_thresholded, [triangle_points], 255)
 
        
        image_closed = cv2.morphologyEx(image_thresholded, cv2.MORPH_CLOSE,
                                kernel=np.ones(x, np.uint8))
        

        # Define the triangle's vertices

        horizonEdge = dy_of_1_above_thresh_hough(image_closed,20)
        
        horizonEdgeDialated = cv2.dilate(horizonEdge,(10,10))
        
        lines = cv2.HoughLines(horizonEdgeDialated,1, np.pi/180,100)

        if plot:
            plt.imshow(masked_image)
            plt.show()
            
            plt.imshow(horizonEdgeDialated)
            xvec = np.array(range(image_closed.shape[1]))
            if np.any(lines):
                for x in lines:
                    plt.plot(xvec,houghLineFunction(-xvec,x[0,0],np.pi/2-x[0,1]))
            plt.show()
            
            
        if np.any(lines):
            avgLine = np.mean(lines,axis=0)      
            medianRho = np.median(lines,axis=0)
            if plot:      
                plt.imshow(horizonEdgeDialated)
                plt.plot(xvec,houghLineFunction(-xvec,medianRho[0,0],np.pi/2-avgLine[0,1]))
                plt.show()
            return avgLine, True
        return np.array([[np.nan, np.nan]]), False
    
def dy_of_1_above_thresh(img, polygon_coords,xVec):
    # Compute the differences in the y-direction
    dy = np.diff(img, axis=0, append=0)
    
    # Create a Polygon object from the provided coordinates
    polygon = Polygon(polygon_coords)
    # Initialize the result array
    result = np.zeros(img.shape[1])
    # Loop through each column of the image
    for col in range(dy.shape[1]):
        dy_pos = np.flip(np.ravel(np.where(dy[:, col] > 0)))  # Find where the change is positive
        
        if len(dy_pos) == 0:
            continue
        # Loop through the points found and check if any are inside the polygon
        
        for idx in range(len(dy_pos)):
            point = Point(xVec[col], dy_pos[idx])
            if not polygon.contains(point):
                result[col] = dy_pos[idx]
                break
    return result

def dy_of_1_above_thresh_hough(img,thresh):
    dy = np.diff(img,axis=0,append=0)
    result = img.copy()
    result.fill(0)
    for col in range(dy.shape[1]):
        dy_pos = np.ravel(np.where(dy[:,col] > 0))
        if len(dy_pos)==0:
            continue
        dy_pos_under_thresh_ind = np.ravel(np.where(dy_pos < img.shape[0]-thresh))
        if len(dy_pos_under_thresh_ind)==0:
            continue
        result[dy_pos[dy_pos_under_thresh_ind[-1]],col] = 255
    return result

def draw_hough_line_edge(frame, rho, theta, color=(0, 255, 0), thickness=2):
    """
    Draw a line on the given frame using rho and theta from Hough Transform,
    ensuring the line touches the edges of the frame.
    
    Parameters:
        frame (numpy.ndarray): The image on which to draw the line.
        rho (float): Distance from the origin to the line.
        theta (float): Angle in radians of the line's normal vector.
        color (tuple): Line color in BGR format (default is green).
        thickness (int): Thickness of the line (default is 2).
        
    Returns:
        numpy.ndarray: The image with the line drawn.
    """
    height, width = frame.shape[:2]
    
    # Calculate the sin and cos of theta
    a = np.cos(theta)
    b = np.sin(theta)
    
    # x0, y0 is the point on the line closest to the origin
    x0 = a * rho
    y0 = b * rho

    # Calculate intersections with the edges of the image
    points = []

    # Intersection with the top edge (y = 0)
    if b != 0:
        x_top = int((rho - 0 * b) / a)
        if 0 <= x_top <= width:
            points.append((x_top, 0))

    # Intersection with the bottom edge (y = height)
    if b != 0:
        x_bottom = int((rho - height * b) / a)
        if 0 <= x_bottom <= width:
            points.append((x_bottom, height))

    # Intersection with the left edge (x = 0)
    if a != 0:
        y_left = int((rho - 0 * a) / b)
        if 0 <= y_left <= height:
            points.append((0, y_left))

    # Intersection with the right edge (x = width)
    if a != 0:
        y_right = int((rho - width * a) / b)
        if 0 <= y_right <= height:
            points.append((width, y_right))

    # If we found two intersection points, draw the line
    if len(points) == 2:
        cv2.line(frame, points[0], points[1], color, thickness)

def draw_horizon_lines_split_frames_split(image, n_splits, n_lines=10):
    images = np.array_split(image,n_splits,axis=1)
    lines = []
    for i in range(n_splits):
        if (i==(n_splits-1)/2):
            #rho1=False
            rho1,theta1=detect_horizon_line(images[i],n_lines,"center",i*images[i].shape[1])
        else:
            rho1,theta1=detect_horizon_line(images[i],14,"hough",i*images[i].shape[1])
        if rho1:
            rho1 = rho1 + i*images[i].shape[1]*np.cos(theta1)
            lines.append([rho1,theta1])
        lowerB = int(i-1 > 0) * (i-1)
        for j in range(lowerB,i):
            neighbors = np.concatenate([images[j],images[i]],axis=1)
            rho2, theta2 = detect_horizon_line(neighbors,22,"center",j*neighbors.shape[1]//2)
            if rho2:
                rho2 = rho2 + j*neighbors.shape[1]/2*np.cos(theta2)
                lines.append([rho2,theta2])   
    return lines       

def draw_one_line_per_frame(input_video, output_video, ss, frame_max):
    """
    Processes a video frame by frame, draws one detected line on each frame, and saves the output.

    Args:
        input_video (str): Path to the input video file.
        output_video (str): Path to the output video file.
    """
    # Get video dimensions dynamically
    def get_video_dimensions(video_path):
        probe = subprocess.run(
            ['ffprobe', '-select_streams', 'v:0', '-show_entries',
             'stream=width,height', '-of', 'json', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        video_info = json.loads(probe.stdout)
        width = video_info['streams'][0]['width']
        height = video_info['streams'][0]['height']
        return width, height

    width, height = get_video_dimensions(input_video)

    # Open the input video using ffmpeg
    process1 = (
        ffmpeg
        .input(input_video,ss=ss,to=hhmmss_to_seconds(frame_max))
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True)
    )

    # Open the output video using ffmpeg
    process2 = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
        .output(output_video, pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    frame_count = 0  # Counter for the frames

    rho = np.nan
    theta = np.nan
    line = np.array([[np.nan,np.nan]])
    x1, x2, y1, y2 = 0, width-1, 0, height-1
    while True:
        # Read a frame from the input video
        in_bytes = process1.stdout.read(width * height * 3)
        if not in_bytes:
            break

        # Convert raw bytes to NumPy array
        frame0 = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
            )
        
        frame = frame0.copy()

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if False:
            plt.imshow(gray_frame)
            plt.title("grey")
            plt.show()


            print(frame.shape)
            plt.imshow(frame[:,:,1])
            plt.title("green")
            plt.show()
        # Detect edges in the green frame
        n_splits = 1
        lines = draw_horizon_lines_split_frames_split(gray_frame,n_splits,50)
        for line in lines:
            draw_hough_line_edge(frame,line[0],line[1],(255,0,0),2)
        if len(lines)>1:
            npLines = np.array(lines)
            linesDist = perpendicular_distance_lines_to_point(npLines,400,375)
            withinDistThreshIndicies = find_indices_within_mode_threshold(linesDist,5)
            if np.any(withinDistThreshIndicies):
                avgLine = np.mean(npLines[withinDistThreshIndicies],axis=0)
                draw_hough_line_edge(frame,avgLine[0],avgLine[1],(0,255,0),2)

        
        # lines, succ = detect_horizon_line_hough(gray_frame,150)
        # if succ:
        #     draw_hough_line_edge(frame,lines[0,0],lines[0,1],(0,255,0))
        #     line = lines

        data.append({
            "frame":frame_count,
            "theta1":theta,
            "rho1":rho,
            # "theta2":line[0,1],
            # "rho2":line[0,0]
            })
                
        #frame = drawClosed(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
           
        # Write the processed frame to the output video
        process2.stdin.write(frame.astype(np.uint8).tobytes())

        frame_count += 1  # Increment frame counter

    # Close processes
    process2.stdin.close()
    process1.stdout.close()
    process1.wait()
    process2.wait()
    print("Processing complete. The output video has been saved.")

def hhmmss_to_seconds(hhmmss):
    t = datetime.strptime(hhmmss, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second

# Example Usage
input_video = './PumpDetector/GL010095.LRV'
output_video = 'outpuffff.mp4'
output_csv = "horizoffff.csv"

data = []


houghLineFunction = np.vectorize(lambda x, rho, theta: (rho - x * np.cos(theta)) / np.sin(theta))
inverseHoughLineFunction = np.vectorize(lambda y, rho, theta: (y * np.cos(theta) - rho) / np.sin(theta))
starting_second = 0
frame_count = "00:35:14"

draw_one_line_per_frame(input_video, output_video, starting_second, frame_count)

df = pd.DataFrame(data)

df.to_csv(output_csv)


# todo - add weights to the average of rho, lower = highter weight
# if fail, try with lower threshold
