from PIL import Image

im1=Image.open("lena.png")
im2=Image.open("lena_modified.png")

L=512
im_out=Image.new("RGB",(L,L),"white")
for x in range(L):
    for y in range(L):
        if im2.getpixel((x,y))!=im1.getpixel((x,y)):
            im_out.putpixel((x,y),im2.getpixel((x,y)))

im_out.save("ans_two.png")
