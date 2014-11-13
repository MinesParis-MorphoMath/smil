#modules.
import sys
sys.path.append ("/usr/local/lib/Smil")
from smilPython import *
import bpy


#Private funcs
def _offsetToPixel (image, offset):
    width = image.getWidth ()
    height = image.getHeight ()
    return [offset % width, height-(offset % ( width * height )) // width, offset // ( width * height )]

def _pixelToOffset (image, v, origin=[0.0,0.0,0.0]):
    co = [v[0],v[1]]
    width = image.getWidth ()
    height = image.getHeight ()
    max_dim = max (width, height)
    y = v[1]-origin[1]
    co[0] -= origin[0]
    co[0] += width/2
    co[1] = height - y - height/2

    return int (co[0] + co[1]*width)
    
    
label_mats_exist=False
def showLabel (obj, im, origin=[0.0,0.0,0.0]) :
    global label_mats_exist
    colors =  [  [255,29,0],
                [230,255,0],
                [0,255,52],
                [0,178,255],
                [104,0,255],
                [255,0,125],
                [255,155,0],
                [73,255,0],
                [0,255,207],
                [0,22,255],
                [255,0,251]]

    mesh = obj.data

    mat_bg = bpy.data.materials.new ('label_background')
    mat_bg.diffuse_color = [0.0,0.0,0.0]
    mesh.materials.append(mat_bg)
    if not label_mats_exist:
        i=0
        for c in colors:
            mat = bpy.data.materials.new('label_color_'+str(i))
            mat.diffuse_color = [c[0]/255.0,c[1]/255.0,c[2]/255.0]
            mesh.materials.append (mat) 
            i += 1
        label_mats_exist = True
    else:
        for i in range(len(colors)):
            mesh.materials.append (bpy.data.materials['label_color_'+str(i)])
            
    for f in mesh.polygons:
        v = [
(mesh.vertices[f.vertices[0]].co[0]+mesh.vertices[f.vertices[1]].co[0]+mesh.vertices[f.vertices[2]].co[0]+mesh.vertices[f.vertices[3]].co[0])/4,
                (mesh.vertices[f.vertices[0]].co[1]+mesh.vertices[f.vertices[1]].co[1]+mesh.vertices[f.vertices[2]].co[1]+mesh.vertices[f.vertices[3]].co[1])/4
            ]
            
        offset = _pixelToOffset (im, v, origin)
        pixel_value = im.getPixel (offset)

        if pixel_value > 0:
            f.material_index =  ((pixel_value-1) % len(colors)) +1
        else:
            f.material_index = 0
                
                
#Create Topology from 2D-Pictures.
smil_image_index=0
def topography (image, origin=[0.0,0.0,0.0], style='cube'):
    global smil_image_index
                
    def _draw_topology_vertices (width, height, pixel, value, vertices, edges, faces):
        vertices.append ((pixel[0]-width/2.0, pixel[1]-height/2.0, value))     
    def _draw_topology_edges (width, height, pixel, value, vertices, edges, faces):
        vertices.append ((pixel[0]-width/2, pixel[1]-height/2, value))
        vertices.append ((pixel[0]-width/2, pixel[1]-height/2, 0))
        edges.append ([len(vertices)-2,len(vertices)-1])
    def _draw_topology_faces (width, height, pixel, value, vertices, edges, faces):
        vertices.append ((pixel[0]-width/2-0.5, pixel[1]-height/2-0.5, value))
        vertices.append ((pixel[0]-width/2-0.5, pixel[1]-height/2+0.5, value))
        vertices.append ((pixel[0]-width/2+0.5, pixel[1]-height/2+0.5, value))
        vertices.append ((pixel[0]-width/2+0.5, pixel[1]-height/2-0.5, value))
        faces.append ([len(vertices)-4,len(vertices)-3,len(vertices)-2,len(vertices)-1])     
    def _draw_topology_cubes (width, height, pixel, value, vertices, edges, faces):
        vertices.append ((pixel[0]-width/2-0.5, pixel[1]-height/2-0.5, value))
        vertices.append ((pixel[0]-width/2-0.5, pixel[1]-height/2-0.5, 0))
        vertices.append ((pixel[0]-width/2-0.5, pixel[1]-height/2+0.5, value))
        vertices.append ((pixel[0]-width/2-0.5, pixel[1]-height/2+0.5, 0))
        vertices.append ((pixel[0]-width/2+0.5, pixel[1]-height/2+0.5, value))
        vertices.append ((pixel[0]-width/2+0.5, pixel[1]-height/2+0.5, 0))
        vertices.append ((pixel[0]-width/2+0.5, pixel[1]-height/2-0.5, value))
        vertices.append ((pixel[0]-width/2+0.5, pixel[1]-height/2-0.5, 0))        
        faces.append ([len(vertices)-8,len(vertices)-6,len(vertices)-4,len(vertices)-2])
        faces.append ([len(vertices)-6,len(vertices)-5,len(vertices)-7,len(vertices)-8])
        faces.append ([len(vertices)-8,len(vertices)-2,len(vertices)-1,len(vertices)-7])
        faces.append ([len(vertices)-2,len(vertices)-4,len(vertices)-3,len(vertices)-1])
        faces.append ([len(vertices)-4,len(vertices)-6,len(vertices)-5,len(vertices)-3])

    #Set up names of topology in blender context.
    meshName = image.getName()
    if meshName == '':
        meshName = 'smil_' + str(smil_image_index)
        smil_image_index += 1
    objName = meshName + '_obj'
    meshName += '_mesh'
    
    #Create object and corresponding mesh
    mesh = bpy.data.meshes.new (meshName)   
    obj = bpy.data.objects.new (objName, mesh)
    obj.location = origin
        
    #Link in to the active scene
    scn = bpy.context.scene
    scn.objects.link(obj)
    scn.objects.active = obj
    obj.select = True
    
    #Create Mesh data
    nbr_pixel = image.getPixelCount ()
    width = image.getWidth()
    height = image.getHeight()
    func_data = _draw_topology_vertices
    vertices = []
    edges = []
    faces = []
    if style == 'edge':
        func_data = _draw_topology_edges    
    elif style == 'face':
        func_data = _draw_topology_faces
    elif style == 'cube':
        func_data = _draw_topology_cubes
    for offset in range(nbr_pixel):
        func_data(width, height, _offsetToPixel (image, offset), image.getPixel(offset), vertices, edges, faces)    
    mesh.from_pydata (vertices, edges, faces)

    max_dim = max (width, height)

    obj.scale = [2/max_dim,2/max_dim,2/max_dim]
    bpy.ops.object.mode_set (mode='EDIT')
    bpy.ops.mesh.remove_doubles()
    bpy.ops.object.mode_set (mode='OBJECT')

    #Final step
    mesh.update (calc_edges=True)
            
    return obj
    

#create Arrows
smil_arrow_index=0
def arrows (imageTopo, imageArrow, origin=[0.0,0.0,0.0]):
    global smil_arrow_index
    #visual offset in altitude.
    hoffset = 0.1
    def _draw (width, height, pixel, se, arrow_value, height_value, vertices, edges, faces):
        if (arrow_value > 0):
            first_vertice = len(vertices)
            vertices.append ((pixel[0]-width/2.0, pixel[1]-height/2.0, height_value + hoffset))
            for i in range(len(se.points)):
                flag = 1 << i
                if (flag & arrow_value):
                    vertices.append ((pixel[0]-width/2+se.points[i].x/3.0, pixel[1]-height/2.0-se.points[i].y/3.0, height_value + hoffset))
                    edges.append ([first_vertice,len(vertices)-1])
        
    #Set up names of topology in blender context.
    meshName = imageArrow.getName()
    if meshName == '':
        meshName = 'smil_arrow_' + str(smil_arrow_index)
        smil_arrow_index += 1
    objName = meshName + '_obj'
    meshName += '_mesh'
    
    #Create object and corresponding mesh
    mesh = bpy.data.meshes.new (meshName)   
    obj = bpy.data.objects.new (objName, mesh)
    obj.location = origin
        
    #Link in to the active scene
    scn = bpy.context.scene
    scn.objects.link(obj)
    scn.objects.active = obj
    obj.select = True
    
    #Create Mesh data
    nbr_pixel = imageArrow.getPixelCount ()
    width = imageArrow.getWidth()
    height = imageArrow.getHeight()
    func_data = _draw
    vertices = []
    edges = []
    faces = []
    for offset in range(nbr_pixel):
        func_data(width, height, _offsetToPixel (imageArrow, offset), se, imageArrow.getPixel(offset), imageTopo.getPixel(offset), vertices, edges, faces)    
    mesh.from_pydata (vertices, edges, faces)

    max_dim = max (width, height)

    obj.scale = [2/max_dim,2/max_dim,2/max_dim]
    bpy.ops.object.mode_set (mode='EDIT')
    bpy.ops.mesh.remove_doubles()
    bpy.ops.object.mode_set (mode='OBJECT')

    #Final step
    mesh.update ()
    
    return obj  
    
    
def clearScene():
    bpy.ops.object.mode_set (mode='OBJECT')
    bpy.ops.object.select_by_type (type='MESH')
    bpy.ops.object.delete (use_global=False)

#clearScene()



##### Morphological Operations

im = Image ("/home/chabardes/lena_crop.png")
#im = Image ("/usr/local/share/Morph-M/Images/Gray/test8.png")
#im = Image ("http://cmm.ensmp.fr/~faessel/smil/images/lena.png")
imG = Image (im)
imA = Image (im)
imA2 = Image (im)
imM = Image (im)
imL = Image (im)
se = cSE().noCenter()

#gradient (im, imG, se)
copy (im,imG)
fastMinima (imG, imM, se)
labelFast (imM, imL, se)
mask (imG, imM, imM)

arrow (imG, "==", imA, se)
arrow (imG, "<=", imA2, se, 0)


##### Display
objMin = topography (imM, [0.0,0.0,0.0], 'face')
objTopo = topography (imG, [0.0,0.0,0.0])
arrMin = arrows (imG, imA)
arrLow = arrows (imG, imA2)

#Materials.
#labelFast (imM, imM, se)
showLabel (objMin, imL)
objTopo.data.materials.append(bpy.data.materials['Topo'])
#arrPlateau.data.materials.append(bpy.data.materials['Arrows'])
arrMin.data.materials.append(bpy.data.materials['Arrows2'])