import cv2
import numpy as np

class Convolution:
    def __call__(self, in_data:np.ndarray, filters:np.ndarray, B, stride_col = 1, stride_row = 1):

        self.in_data = in_data
        self.filters = filters

        self.stride_col = stride_col
        self.stride_row = stride_row

        self.B = B

        self.H, self.W, self.C = self.in_data.shape 

        self.M, self.C, self.S, self.R = self.filters.shape

        self.O = 0

        output_cols = 1 + (self.W - self.S) // self.stride_col
        output_rows = 1 + (self.H - self.R) //self.stride_row

        for dim in [self.M, output_cols, output_rows]:
            self.O = [self.O]*dim

        self.O = np.array(self.O)

        for x in range(output_rows):
            for y in range(output_cols):
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
                    self.O[x][y][m] += self.B[m] + self.in_data[self.stride_row*x+i][self.stride_col*y+j][k] * self.filters[m][k][i][j]
                    
if __name__ == '__main__':

    orig_img = cv2.imread('port.jpg', cv2.IMREAD_GRAYSCALE)
    img_filters = []
    orig_img = np.expand_dims(orig_img, axis=2)
    print('New shape: ', orig_img.shape)
    C = orig_img.shape[2]
    print('Channels: ', C)
    B = [0] * C
    sharpener = np.array([[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]] * C)
    img_filters.append(sharpener)
    img_filters = np.array(img_filters)
    con = Convolution()
    res = con(orig_img, img_filters, B)
    cv2.imwrite('port_sharpened.jpg', res)
    
    
   
