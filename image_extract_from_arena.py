import cv2
img = cv2.imread(r"saved_image110.jpg")
# img = cv2.imread(r"eyantra photos/saved_image54.jpg")
# img = cv2.imread(r"eyantra photos/saved_image99.jpg")
im1 = img[135:215,201:284]
im2 = img[473:558,681:763]
im3 = img[478:551,185:266]
im4 = img[681:762,673:754]
im5 = img[881:963,198:281]

# im2 = img[0:100,100:200]
# cv2.imshow("arena",img)
# cv2.imshow("im1",im1)
# cv2.imshow("im2",im2)
# cv2.imshow("im3",im3)
# cv2.imshow("im4",im4)
# cv2.imshow("im5",im5)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("nim1.jpg",im1)
cv2.imwrite("nim2.jpg",im2)
cv2.imwrite("nim3.jpg",im3)
cv2.imwrite("nim4.jpg",im4)
cv2.imwrite("nim5.jpg",im5)

# im1 = img[128:216,200:285]
# im2 = img[470:558,678:763]
# im3 = img[465:552,179:266]
# im4 = img[678:762,670:754]
# im5 = img[879:963,197:281]

##############################3
# im1 = img[135:215,201:284]
# im2 = img[473:558,681:763]
# im3 = img[471:551,183:266]
# im4 = img[681:762,673:754]
# im5 = img[881:963,198:281]

#lmao
# im1 = img[137:215,205:284]
# im2 = img[476:550,684:755]
# im3 = img[471:549,185:266]
# im4 = img[681:762,673:754]
# im5 = img[881:963,198:281]
#lmao end
