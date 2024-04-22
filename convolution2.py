import cv2
import numpy as np

class Convolution:
    def __call__(self, in_data:np.ndarray, filters:np.ndarray):

        self.in_data = in_data
        self.filters = filters

        self.H, self.W, self.C = self.in_data.shape 

        self.M, self.C, self.S, self.R = self.filters.shape

        self.O = 0

        for dim in [self.M, self.W, self.H]:
            self.O = [self.O]*dim

        self.O = np.array(self.O)

        for x in range(self.H - self.R + 1):
            for y in range(self.W - self.S + 1):
                for m in range(self.M):
                    self.compute_convolution_for_point(m, x, y)
        
        return self.O

    def compute_convolution_for_point(self, m, x, y):
        """
        Вычисление значения для заданных x, y, m
        """

        for i in range(self.R):
            for j in range(self.S):
                for k in range(self.C):
                    self.O[x][y][m] += self.in_data[x+i][y+j][k] * self.filters[m][k][i][j]
                    
if __name__ == '__main__':

    orig_img = cv2.imread('port.jpg', cv2.IMREAD_GRAYSCALE)
    img_filters = []
    orig_img = np.expand_dims(orig_img, axis=2)
    print('New shape: ', orig_img.shape)
    C = orig_img.shape[2]
    print('Channels: ', C)
    sharpener = np.array([[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]] * C)
    img_filters.append(sharpener)
    img_filters = np.array(img_filters)
    con = Convolution()
    res = con(orig_img, img_filters)
    cv2.imwrite('port_sharpened.jpg', res)
    
   