import pathlib2
import numpy as np
import json
import random
import requests
from io import BytesIO
from math import trunc
import cv2
import re


# Load the dataset json
class LiveCellDataset:
    def __init__(self, annotation_path, image_dir):
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.colors = ['limegreen', 'blue', 'purple', 'red', 'green', 'orange', 'salmon', 'pink',
                       'gold', 'orchid', 'slateblue', 'seagreen', 'darkgreen', 'olive',
                       'teal', 'aquamarine', 'steelblue', 'powderblue', 'dodgerblue', 'navy',
                       'magenta', 'sienna', 'maroon']

        with open(self.annotation_path) as json_file:
            self.livecell = json.load(json_file)

        self.process_info()
        self.process_licenses()
        self.process_categories()
        self.process_images()
        self.process_segmentations()

    def display_info(self):
        print('Dataset Info:')
        print('=============')
        if self.info is None:
            return
        for key, item in self.info.items():
            print('  {}: {}'.format(key, item))

        requirements = [['description', str],
                        ['url', str],
                        ['version', str],
                        ['year', str],
                        ['contributor', str],
                        ['date_created', str]]
        for req, req_type in requirements:
            if req not in self.info:
                print('ERROR: {} is missing'.format(req))
            elif type(self.info[req]) != req_type:
                print('ERROR: {} should be type {}'.format(req, str(req_type)))
        print('')

    def display_licenses(self):
        print('Licenses:')
        print('=========')

        if self.licenses is None:
            return
        requirements = [['id', int],
                        ['url', str],
                        ['name', str]]
        for license in self.licenses:
            for key, item in license.items():
                print('  {}: {}'.format(key, item))
            for req, req_type in requirements:
                if req not in license:
                    print('ERROR: {} is missing'.format(req))
                elif type(license[req]) != req_type:
                    print('ERROR: {} should be type {}'.format(
                        req, str(req_type)))
            print('')
        print('')

    def display_categories(self):
        print('Categories:')
        print('=========')
        for sc_key, sc_val in self.super_categories.items():
            print('  super_category: {}'.format(sc_key))
            for cat_id in sc_val:
                print('    id {}: {}'.format(
                    cat_id, self.categories[cat_id]['name']))
            print('')

    def display_image(self, image_id, show_polys=True, show_bbox=True, show_crowds=True, use_url=False, use_cv=True,
                      verbose=False):
        """
        display the image and the corresponding annotations according to the given image_id or the one at random
        :param image_id: str image id given or at random
        :param show_polys: boolean flag if polygons should be visualized
        :param show_bbox: boolean flag if bounding box should be visualized
        :param show_crowds: boolean flag if the crowd (a cloud of objects) should be visualized
        :param use_url: boolean flag if url should be leveraged
        :param use_url: boolean flag if opencv should be leveraged for visualization
        :param verbose: boolean flag if the additional information should be shown in the terminal
        :return:
        """
        print('Image:')
        print('======')
        if image_id == 'random':
            image_id = random.choice(list(self.images.keys()))

        # Print the image info
        image = self.images[image_id]
        for key, val in image.items():
            print('  {}: {}'.format(key, val))

        # Open the image through url
        if use_url:
            image_path = image['url']
            response = requests.get(image_path)
            image = PILImage.open(BytesIO(response.content))
            if verbose:
                print("It successfully opens the image via url.")
        # Open the image through the local path
        else:
            # filtering for the cell type dir is necessary due to the additional dir structure for cell type
            # image_path = "{}/{}".format(self.image_dir, image['file_name'])
            filtered_dir_obj = re.match(r"[a-zA-Z0-9]+", image['file_name'], flags=0)
            filtered_dir = filtered_dir_obj.group(0)
            image_path = self.image_dir / filtered_dir / image['file_name']
            if use_cv:
                image = cv2.imread(image_path.as_posix())
            else:
                image = PILImage.open(image_path.as_posix())
            if verbose:
                print("It successfully opens the image via local path.")

        # Calculate the size and adjusted display size with aspect ratio being the same
        max_width = 704
        image_width, image_height = image.size
        adjusted_width = min(image_width, max_width)
        adjusted_ratio = adjusted_width / image_width
        adjusted_height = adjusted_ratio * image_height

        # Create list of polygons to be drawn
        polygons = {}
        bbox_polygons = {}
        rle_regions = {}
        poly_colors = {}
        bbox_categories = {}
        # Print the annotation info for the specific image id
        print('  segmentations ({}),'.format(
            len(self.segmentations[image_id])))
        current_img_one_annot = self.segmentations[image_id][0]
        print('  and details for image_id {}:'.format(current_img_one_annot['image_id']))
        for i, segm in enumerate(self.segmentations[image_id]):
            polygons_list = []
            if segm['iscrowd'] != 0:
                # Gotta decode the RLE (not adapted presently!)
                px = 0
                x, y = 0, 0
                rle_list = []
                for j, counts in enumerate(segm['segmentation']['counts']):
                    if j % 2 == 0:
                        # Empty pixels
                        px += counts
                    else:
                        # Need to draw on these pixels, since we are drawing in vector form,
                        # we need to draw horizontal lines on the image
                        x_start = trunc(
                            trunc(px / image_height) * adjusted_ratio)
                        y_start = trunc(px % image_height * adjusted_ratio)
                        px += counts
                        x_end = trunc(trunc(px / image_height)
                                      * adjusted_ratio)
                        y_end = trunc(px % image_height * adjusted_ratio)
                        if x_end == x_start:
                            # This is only on one line
                            rle_list.append(
                                {'x': x_start, 'y': y_start, 'width': 1, 'height': (y_end - y_start)})
                        if x_end > x_start:
                            # This spans more than one line
                            # Insert top line first
                            rle_list.append(
                                {'x': x_start, 'y': y_start, 'width': 1, 'height': (image_height - y_start)})

                            # Insert middle lines if needed
                            lines_spanned = x_end - x_start + 1  # total number of lines spanned
                            full_lines_to_insert = lines_spanned - 2
                            if full_lines_to_insert > 0:
                                full_lines_to_insert = trunc(
                                    full_lines_to_insert * adjusted_ratio)
                                rle_list.append(
                                    {'x': (x_start + 1), 'y': 0, 'width': full_lines_to_insert, 'height': image_height})

                            # Insert bottom line
                            rle_list.append(
                                {'x': x_end, 'y': 0, 'width': 1, 'height': y_end})
                if len(rle_list) > 0:
                    rle_regions[segm['id']] = rle_list
            else:
                # Add one polygon for an object
                for segmentation_points in segm['segmentation']:
                    segmentation_points = np.multiply(
                        segmentation_points, adjusted_ratio).astype(int)  # return np.array([[...]]) (1*m)
                    polygons_list.append(
                        str(segmentation_points).lstrip('[').rstrip(']'))
            polygons[segm['id']] = polygons_list
            # the following snippet for traversing the self.colors list is unnecessary since there is only one kind
            # of segmentation label for each image
            # if i < len(self.colors):
            #     poly_colors[segm['id']] = self.colors[i]
            # else:
            #     poly_colors[segm['id']] = 'white'
            poly_colors[segm['id']] = self.colors[0]
            bbox = segm['bbox']
            bbox_points = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1],
                           bbox[0] + bbox[2], bbox[1] +
                           bbox[3], bbox[0], bbox[1] + bbox[3],
                           bbox[0], bbox[1]]
            bbox_points = np.multiply(bbox_points, adjusted_ratio).astype(int)
            bbox_polygons[segm['id']] = str(
                bbox_points).lstrip('[').rstrip(']')
            bbox_categories[segm['id']] = self.categories[segm['category_id']]
            # Print details
            if verbose:
                print('    {}:{}:{}'.format(
                    segm['id'], poly_colors[segm['id']], self.categories[segm['category_id']]))

        # Draw segmentation polygons on image
        html = '<div class="container" style="position:relative;">'
        html += '<img src="{}" style="position:relative;top:0px;left:0px;width:{}px;">'.format(
            image_path, adjusted_width)
        html += '<div class="svgclass"><svg width="{}" height="{}">'.format(
            adjusted_width, adjusted_height)

        if show_polys:
            for seg_id, points_list in polygons.items():
                fill_color = poly_colors[seg_id]
                stroke_color = poly_colors[seg_id]
                for points in points_list:
                    html += '<polygon points="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0.5" />'.format(
                        points, fill_color, stroke_color)

        if show_crowds:
            for seg_id, rect_list in rle_regions.items():
                fill_color = poly_colors[seg_id]
                stroke_color = poly_colors[seg_id]
                for rect_def in rect_list:
                    x, y = rect_def['x'], rect_def['y']
                    w, h = rect_def['width'], rect_def['height']
                    html += '<rect x="{}" y="{}" width="{}" height="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0.5; stroke-opacity:0.5" />'.format(
                        x, y, w, h, fill_color, stroke_color)

        if show_bbox:
            for seg_id, points in bbox_polygons.items():
                x, y = [int(i) for i in points.split()[:2]]
                html += '<text x="{}" y="{}" fill="yellow">{}</text>'.format(
                    x, y, bbox_categories[seg_id]["name"])
                fill_color = poly_colors[seg_id]
                stroke_color = poly_colors[seg_id]
                html += '<polygon points="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0" />'.format(
                    points, fill_color, stroke_color)

        html += '</svg></div>'
        html += '</div>'
        html += '<style>'
        html += '.svgclass { position:absolute; top:0px; left:0px;}'
        html += '</style>'
        return html

    def process_info(self):
        self.info = self.livecell.get('info')

    def process_licenses(self):
        self.licenses = self.livecell.get('licenses')

    def process_categories(self):
        self.categories = {}
        self.super_categories = {}
        for category in self.livecell['categories']:
            cat_id = category['id']
            super_category = category['supercategory']
            cat_name = category['name']

            # Add category to the categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
            else:
                print("ERROR: Skipping duplicate category id: {}".format(category))

            # Add category to super_categories dict
            if super_category not in self.super_categories:
                # Create a new set with the category id
                self.super_categories[super_category] = {cat_id}
            else:
                self.super_categories[super_category] |= {
                    cat_id}  # Add category id to the set

    def process_images(self):
        # create the dict with key (str): image_id and value (dict): each row of key-value pairs of image metadata
        self.images = {}
        for image in self.livecell['images']:
            image_id = image['id']
            if image_id in self.images:
                print("ERROR: Skipping duplicate image id: {}".format(image))
            else:
                self.images[image_id] = image

    def process_segmentations(self):
        # create the dict with key (str): image_id and value (list): list of annotation metadata (dict)
        self.segmentations = {}
        for segmentation in self.livecell['annotations'].values():
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)
