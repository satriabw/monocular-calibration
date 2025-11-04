# https://github.com/AubreyC/trajectory-extractor/blob/master/traj_ext/tracker/cameramodel.py
import numpy as np
import cv2

class CameraModel:
    def __init__(self, camera_matrix, rot_matrix, tvec, dist_coeffs, homography=None):
        self.camera_matrix = camera_matrix
        self.rot_matrix = rot_matrix
        self.dist_coeffs = dist_coeffs
        self.homography = homography

        self.use_homography = False
        if self.homography:
            self.use_homography = True

        self.rvec          = cv2.Rodrigues(self.rot_matrix)[0]                       # 3x1
        self.tvec          = tvec
    
    def project_points_homography(self, pixel_coord):
        # Use homography if available
        H_inv = np.linalg.inv(self.homography)
        pixel_homog = np.vstack((pixel_coord.T, np.ones((1, pixel_coord.shape[0]))))
        ground_coords = H_inv @ pixel_homog
        ground_coords /= ground_coords[2, :]  # Normalize by the third coordinate
        return ground_coords[:2, :].T
    
    def backproject_points_homography(self, point_xy):
        point_homog = np.array([point_xy[0], point_xy[1], 1.0]).reshape(3,1)
        img_coords = self.homography @ point_homog
        img_coords /= img_coords[2,0]
        u, v = img_coords[0,0], img_coords[1,0]
        return (float(u), float(v))

    def project_to_ground(self, pixel_coord):
        """
        Project 3D points to ground plane using camera parameters.
        
        Args:
            pixel_coords: 2D pixel coordinates in the image plane

        Returns:
            2D points on the ground plane
        """
        if self.use_homography:
            return self.project_points_homography(pixel_coord)

        # Build Homography using camera intrinsics and extrinsics to simplify projection
        # https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/amod/Lecture_6/20191007%20-%20ETH%20-%2005%20-%20CV%20IV%20-%20Camera%20calibration.pdf
        # Assuming the ground plane is at Z=0 in camera coordinates
        pixel_coord = pixel_coord.T
        r1, r2 = self.rot_matrix[:,0:1], self.rot_matrix[:,1:2]
        tvec_reshaped = self.tvec.reshape(-1, 1)
        H = self.camera_matrix @ np.hstack((r1, r2, tvec_reshaped))

        # Pixel to ground XY (assuming Z=0)
        Hinbv = np.linalg.inv(H)
        uv1 = np.vstack((pixel_coord, np.ones((1, pixel_coord.shape[1]))))
        ground_coords = Hinbv @ uv1
        ground_coords /= ground_coords[2, :] # Dividing by W

        return ground_coords[:2, :]
    
    def project_point(self, point_xy):
        """
        Project ONE 2D point from ground plane (X,Y) to image pixels (u,v).
        Assumes Z=0 (point is on the ground plane).
        """
        if self.use_homography:
            return self.backproject_points_homography(point_xy)
  
        # Append Z=0 to the XY coordinates
        point_xyz = np.append(point_xy, 0).reshape(1,1,3)
        
        img_pts, _ = cv2.projectPoints(point_xyz, self.rvec, self.tvec.reshape(3,1),
                                      self.camera_matrix, self.dist_coeffs)
        u, v = img_pts[0,0,0], img_pts[0,0,1]
        return (float(u), float(v))

    def save_to_yml(self, filepath):
        wfs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
        wfs.write("camera_matrix", self.camera_matrix)
        wfs.write("dist_coeffs", self.dist_coeffs)
        wfs.write("rot_matrix", self.rot_matrix)
        wfs.write("tvec", self.tvec)
        if self.homography:
            wfs.write("homography", self.homography)

        wfs.release()

    @classmethod
    def load_from_yml(cls,filepath):
        fr = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
        homography=fr.getNode("homography").mat() if fr.getNode("homography") else None
        camera_matrix = fr.getNode("camera_matrix").mat()
        dist_coeffs = fr.getNode("dist_coeffs").mat()
        rot_matrix = fr.getNode("rot_matrix").mat()
        tvec = fr.getNode("tvec").mat()
        fr.release()

        return cls(
            camera_matrix=camera_matrix,
            rot_matrix=rot_matrix,
            tvec=tvec,
            dist_coeffs=dist_coeffs,
            homography=homography
        )
