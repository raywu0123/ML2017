from PIL import Image
import sys
im1=Image.open(sys.argv[1])
im2=Image.open(sys.argv[2])

L=512
im_out=Image.new("RGB",(L,L),"white")
for x in range(L):
    for y in range(L):
        if im2.getpixel((x,y))!=im1.getpixel((x,y)):
            im_out.putpixel((x,y),im2.getpixel((x,y)))

im_out.save("ans_two.png")
