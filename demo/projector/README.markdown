# Lighthouse: Projector

Let's setup a scene including a camera (frustum lower left corner), a point light source (yellow circle), a projector (frustum above yellow circle) and a complex 3D object (fox head) with a plane in the background.

![](images/scene.png)

The image we project with the projector is this

![](images/image.png)

The result looks pretty convincing. The projected image is distorted by the surface structure of the fox head. The darkest parts of the image behind the fox head are only illuminated by ambient lighting, neither the projector nor the point light reach these areas. The brighter gray are areas illuminated by the point light but not the projector.

![](images/projection.png)

