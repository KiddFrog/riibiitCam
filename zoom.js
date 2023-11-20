const imageFolder = '/Users/domm/Desktop/froggo cam/riibiitCam/frames/image_1.gif';
const imageName = 'image.jpg';

const loadImage = () => {
  const imgElement = document.createElement('img');
  imgElement.src = `${imageFolder}${imageName}`;
  document.body.appendChild(imgElement);
};

loadImage();