import torch
import pydiffvg
import svgpathtools

# currently these functions 
def canonizePaths(paths_tensor: torch.Tensor, 
                  im_height: int=224, 
                  im_width: int=224,
                  num_paths: int=16,
                  num_control_points: int=4) -> torch.Tensor:
    center_x = im_width / 2
    center_y = im_height / 2
    # reshape paths_tensor 
    paths_tensor = paths_tensor.reshape(-1, 2)
    # center the paths_tensor
    paths_tensor = paths_tensor - torch.tensor([center_x, center_y])
    # normalize the paths_tensor
    paths_tensor = paths_tensor / torch.tensor([center_x, center_y])
    return paths_tensor.reshape(-1, num_paths * num_control_points * 2)

def decanonizePaths(paths_tensor: torch.Tensor, 
                  im_height: int=224, 
                  im_width: int=224,
                  num_paths: int=16,
                  num_control_points: int=4) -> torch.Tensor:
    center_x = im_width / 2
    center_y = im_height / 2
    # reshape paths_tensor 
    paths_tensor = paths_tensor.reshape(-1, 2)
    # normalize the paths_tensor
    paths_tensor = paths_tensor * torch.tensor([center_x, center_y])
    # center the paths_tensor
    paths_tensor = paths_tensor + torch.tensor([center_x, center_y])
    return paths_tensor.reshape(-1, num_paths * num_control_points * 2)

def tensor2SVG(paths_tensor: torch.Tensor, 
               s_width: float=1.5, 
               s_color: list=[0.0, 0.0, 0.0, 1.0]) -> dict:
    # reshape the tensor to [dont care, 4, 2]
    paths_tensor = paths_tensor.reshape(-1, 4, 2)
    num_paths = paths_tensor.shape[0]
    shapes=[]
    shape_groups=[]
    for i in range(num_paths):
        path_points = paths_tensor[i]
        diffvg_path = pydiffvg.Path(num_control_points = torch.zeros(1, dtype = torch.int32) + 2,
                                    points = path_points,
                                    stroke_width = torch.tensor(s_width),
                                    is_closed = False)
        stroke_color = torch.tensor(s_color)
        shapes.append(diffvg_path)
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                            fill_color = None,
                                            stroke_color = stroke_color)
        shape_groups.append(path_group)
    return {"shapes": shapes, "shape_groups": shape_groups, "paths_tensor": paths_tensor}

def loadClipassoSVG(svg_path: str) -> dict:
    paths, _ = svgpathtools.svg2paths(svg_path)
    # convert to diffVG format to create new SVG
    shapes=[]
    shape_groups=[]
    paths_tensor=[]
    for path in paths:
        # parse each path into 4 2D control points in a torch tensor
        path_points = []
        for segment in path:
            if isinstance(segment, svgpathtools.path.CubicBezier):
                path_points.append([segment.start.real, segment.start.imag])
                path_points.append([segment.control1.real, segment.control1.imag])
                path_points.append([segment.control2.real, segment.control2.imag])
                path_points.append([segment.end.real, segment.end.imag])
            elif isinstance(segment, svgpathtools.path.Line):
                path_points.append([segment.start.real, segment.start.imag])
                path_points.append([segment.end.real, segment.end.imag])
            else:
                raise ValueError("Unsupported path segment type")
        path_points = torch.tensor(path_points)
        paths_tensor.append(path_points)
        diffvg_path = pydiffvg.Path(num_control_points = torch.zeros(1, dtype = torch.int32) + 2,
                                    points = path_points,
                                    stroke_width = torch.tensor(1.5),
                                    is_closed = False)
        stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
        shapes.append(diffvg_path)
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                            fill_color = None,
                                            stroke_color = stroke_color)
        shape_groups.append(path_group)
    paths_tensor = torch.stack(paths_tensor)
    return {"shapes": shapes, "shape_groups": shape_groups, "paths_tensor": paths_tensor}

def saveClipassoSVG(svg_path: str, 
                    shapes: list, 
                    shape_groups: list, 
                    im_height: int=224, 
                    im_width: int=224) -> None:
    pydiffvg.save_svg(svg_path, im_height, im_width, shapes, shape_groups)

def renderCLipassoSVG(shapes: list, 
                      shape_groups: list, 
                      im_height: int=224, 
                      im_width: int=224) -> torch.Tensor:
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(im_width, im_height, shapes, shape_groups)
    img = _render(im_width, # width
                im_height, # height
                2,   # num_samples_x
                2,   # num_samples_y
                0,   # seed
                None,
                *scene_args)
    return img