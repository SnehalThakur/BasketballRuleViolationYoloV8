from simple_image_download import simple_image_download as simp
response = simp.simple_image_download

keywords = ['basket ball']

for kw in keywords:
    response().download(kw, 100)

print("Download Completed")