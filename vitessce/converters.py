class Cells:

  """
  Generic Cells class for constructing the json needed for client side rendering of cell segmentations/scatterplots (UMAP, PCA etc.).

  :param json The json resulting from various calls to add_mapping, add_polygon_outline etc. that can be served to the client.
  """

  def __init__(self, cell_ids=[]):
    """
    Constructor method

    :param list cell_ids: A list of cell ids to be shown in Vitessce.  The order of these will be used to determine the order of future additions to this class, like segmentations and scatterplot coordinates.
    """
    self._cell_ids = cell_ids
    self.json = dict(zip(cell_ids, [{} for _ in cell_ids]))

  def add_mapping(self, name, coords):
    """
    Add a (dimensionality reduction) scatterplot mapping to each cell.

    :param str name: The unique identifier for the mapping, like UMAP, tSNE or PCA.
    :param list coords: A list of lists like [[1, 2], [3, 4], ...] in the order of cell_ids for each cell to be mapped to a scatterplot coorindate.
    """
    for idx, id in enumerate(self._cell_ids):
      if 'mappings' not in self.json[id]:
        self.json[id]['mappings'] = { name: coords[idx] }
      else:
        self.json[id]['mappings'][name] = coords[idx]
  
  def add_xy(self, xy):
    """
    Add a centroid for a spatial segmentation outline to each cell.

    :param list xy: A list of lists like [[1, 2], [3, 4], ...] in the order of cell_ids for each cell to be mapped to a centroid coorindate.
    """
    for idx, id in enumerate(self._cell_ids):
      self.json[id]['xy'] = xy[idx]


  def add_polygon_outline(self, polygon_outline):
    """
    Add a polygon for a spatial segmentation outline to each cell.

    :param list polygon_outline: A list of lists of lists like [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]...] in the order of cell_ids for each cell to be mapped to its segmentation.
    """
    for idx, id in enumerate(self._cell_ids):
      if type(polygon_outline[idx]) != list or any([len(coord) != 2 or type(coord) != list for coord in polygon_outline[idx]]):
        raise TypeError('Polygon outline should be a list of two element lists')
      self.json[id]['poly'] = polygon_outline[idx]

class CellSets:

  """
  Generic CellSets class for constructing the json needed for client side rendering of the cell sets.

  :param json The json resulting from various calls to add_node that can be served to the client.
  """


  def __init__(self, first_node_name):
    """
     Constructor method

    :param str first_node_name: Name of the first node to be added to the tree.
    """

    self.json = {
        "datatype": "cell",
        "version": "0.1.2",
        "tree": [{
            "name": first_node_name,
            "children": []
        }]
    }

  def add_node(self, name, parent_path, cell_set=None):
    """
    Add a node to a parent node.

    :param str name: Name for the new node
    :param list parent_path: List of strings representing the internal nodes to traverse to reach the desired parent node to which we will add the new node, like ['epithelial', 'meso-epithelial']
    :param list cell_set: List of cell ids which will be added to the new node as part of the set.
    """
    parent_node = self._tree_find_node_by_path(parent_path)
    new_node = { "name": name }
    if cell_set:
      new_node['set'] = cell_set
    if 'children' not in parent_node:
      parent_node['children'] = [new_node]
    else:
      parent_node['children'].append(new_node)
  
  def _find_node_by_path(self, node, path, curr_index):
    curr_node_name = path[curr_index]
    if node['name'] == curr_node_name:
      if curr_index == len(path) - 1:
        return node
      if 'children' in node:
        found_nodes = [
          self._find_node_by_path(child, path, curr_index + 1) for child in node['children']
        ]
        found_nodes_not_none = [n for n in found_nodes if n]
        if len(found_nodes_not_none) == 1:
          return found_nodes[0]
    return None

  def _tree_find_node_by_path(self, path):
    found_nodes = [self._find_node_by_path(node, path, 0) for node in self.json['tree']]
    found_nodes_not_none = [n for n in found_nodes if n]
    if len(found_nodes_not_none) == 1:
      return found_nodes_not_none[0]
    return None

class Molecules():

  """
  Generic Molecules class for constructing the json needed for client side rendering of spot data.

  :param json The json resulting from various calls to add_molecule.
  """

  def __init__(self):
    """
    Constructor method
    """
    self.json = {}

  def add_molecule(self, name, coords):
    """
    Add a moleuls to a parent node.

    :param str name: Name for the new molecules
    :param list coords: A list of lists like [[1, 2], [3, 4], ...] or [[1, 2, 3], [3, 4, 5], ...] which denote where in xy space the spot data should be placed for the desired name.
    """
    self.json[name] = coords