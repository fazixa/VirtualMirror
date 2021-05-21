from irisSeg import irisSeg
import matplotlib.pyplot as plt

coord_iris, coord_pupil, output_image = irisSeg('file2.jpg', 40, 70)
print(coord_iris) # radius and the coordinates for the center of iris 
print(coord_pupil) # radius and the coordinates for the center of pupil 
plt.imshow(output_image)
plt.show()