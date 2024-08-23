ffmpeg \
-framerate 10 \
-i "recon/images/%3d0.jpg" \
-vcodec libx264 \
-crf 18 \
-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
"recon/recon.mp4" -y