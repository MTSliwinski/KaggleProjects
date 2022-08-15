import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image


class ImageReshape:
    def __init__(self, filepath:str,
                       sample_number:int,
                       desired_size:tuple,
                       is_grayscale:bool = False,
                       fileformat: str = '.jpg'
                       ):
        self.filepath = filepath
        self.fileformat = fileformat
        self.sample_number = sample_number
        self.desired_size = desired_size
        self.is_grayscale = is_grayscale

    def save_transformed_data(self, desiredpath):
        transform = T.Resize((256, 256))
        transform_grayscale = T.Grayscale()
        transform_totensor = T.Compose([T.ToTensor()])
        for x in range(1,self.sample_number):
            try:
                img = Image.open(self.filepath +f'/{x}' + self.fileformat)
                transformed_img = transform(img)
                if self.is_grayscale:
                    grayscale_img = transform_grayscale(transformed_img)
                    tensor = transform_totensor(grayscale_img)
                    save_image(tensor, desiredpath + f'/{x}' + self.fileformat)
                else:
                    tensor = transform_totensor(transformed_img)
                    save_image(tensor, desiredpath + f'/{x}' + self.fileformat)
            except Exception:
                pass


if __name__ == "__main__":
    reshaper = ImageReshape('tshirt', 2000, (48, 48), is_grayscale=True)
    reshaper.save_transformed_data('tshirt_resized')
