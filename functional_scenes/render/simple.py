from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    FoVPerspectiveCameras,
    look_at_view_transform, RasterizationSettings, MeshRenderer, HardPhongShader,
    SoftGouraudShader, SoftPhongShader, PointLights, BlendParams, MeshRasterizer
)

from . render import AbstractGraphics

ambient_c = ((0.3, 0.3, 0.3),)
diffuse_c = ((0.3, 0.3, 0.3),)
specular_c = ((0.2, 0.2, 0.2),)
default_light = PointLights(ambient_color=ambient_c,
                            diffuse_color=diffuse_c,
                            specular_color=specular_c,
                            location=[[0,3,10]]
                            )
# default_light = PointLights(location=[[0,0,-3]])

class SimpleGraphics(AbstractGraphics):

    def __init__(self, img_size, device,
                 lighting = default_light,
                 shader = SoftPhongShader):
        self.cameras = FoVPerspectiveCameras(device=device,
                                             fov = 37.6,
                                             aspect_ratio = 1.21
                                             )
        lighting.to(device)
        self.lights = lighting
        self.device = device

        raster_settings = RasterizationSettings(
            image_size=img_size,
            bin_size=0, # naive rasterization
            blur_radius=0.00001,
            faces_per_pixel=10,
            cull_backfaces=True
        )

        blend_params = BlendParams(background_color=(0,0,0))

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras,
                                      raster_settings=raster_settings),
            shader=shader(device=device,
                          cameras=self.cameras,
                          lights=self.lights,
                          blend_params=blend_params
                          )
            )

    def set_from_scene(self,scene):
        # Camera
        self.set_camera(scene['camera'])
        self.set_lighting(scene['lights'])

    def set_camera(self, cam_params):
        eye = cam_params['position']
        R, T = look_at_view_transform(device=self.device,
                                      # dist = 19,
                                      at = ((0,0,1),),
                                      eye = [eye])
                                      # elev = -80,
                                      # azim = 0)
        self.cameras.get_camera_center(R=R, T=T) # updates R and T for cameras

    def set_lighting(self, lights):
        pass

    def render(self, mesh):
        out = self.renderer(mesh, cameras=self.cameras)
        return out

