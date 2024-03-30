from roboflow import Roboflow
from PIL import Image
rf = Roboflow(api_key="IaWy30TbFdzY7d4uL4e5")
project = rf.workspace().project("cv-cnfd4")
model = project.version(1).model

# infer on a local image
results = model.predict("data/bb1.jpg", confidence=40, overlap=30).json()

print(results)
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('resultBB1.jpg')
# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())