from smilPython import *


## Custom mosaicToGraph functor which takes the mean pixel value along the border between two basins instead of the min value
class customMosaicToGraphFunct(mosaicToGraphFunct_UINT16_UINT16):
    def initialize(self, imMos, dummy, se):
        mosaicToGraphFunct_UINT16_UINT16.initialize(self, imMos, dummy, se)
        self.edges = self.graph.getEdges()
        self.borderValues = []
        return RES_OK

    def processPixel(self, i, relOffList):
        pixSum = 0.0
        curVal = self.imMosaic[i]
        for nb in relOffList:
            nbVal = self.imMosaic[i + nb]
            if curVal != nbVal:
                # check if a corresponding edge already exists
                eInd = self.graph.findEdge(curVal, nbVal)
                if eInd == -1:  # not found -> create a new one
                    if type(self.imNodeValues) != type(
                        None
                    ):  # If an image with node values is given
                        self.graph.addNode(curVal, self.imNodeValues[i])
                        self.graph.addNode(nbVal, self.imNodeValues[i + nb])
                    if type(self.imEdgeValues) != type(
                        None
                    ):  # If an image with edge values is given
                        self.borderValues.append([self.imEdgeValues[i]])
                    self.graph.addEdge(
                        curVal, nbVal, 0, False
                    )  # False -> avoids to check if the edge already exists (done previoulsy)
                else:  # edge already exists
                    if type(self.imEdgeValues) != type(None):
                        self.borderValues[eInd].append(self.imEdgeValues[i])

    def finalize(self, imMos, dummy, se):
        mosaicToGraphFunct_UINT16_UINT16.finalize(self, imMos, dummy, se)
        for i in range(len(self.borderValues)):
            values = self.borderValues[i]
            self.edges[i].weight = sum(values) / len(values)
        return RES_OK


im1 = Image("https://smil.cmm.minesparis.psl.eu/images/mosaic.png")
im2 = Image(im1)
im3 = Image(im1)
imMos = Image(im1, "UINT16")
imArea = Image(imMos)
imSeg = Image(imMos)

label(im1, imMos)
labelWithArea(im1, imArea)

func = customMosaicToGraphFunct()
g = func(imMos, imArea)

drawGraph(imMos, g, imSeg)
imMos.getViewer().drawOverlay(imSeg)

g.removeNodeEdges(3)
graphToMosaic(imMos, g, imSeg)

im1.show()
imMos.showLabel()
imSeg.showLabel()
