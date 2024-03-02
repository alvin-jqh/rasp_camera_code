import numpy as np

class object_matcher:
    def __init__(self, a=400, b=20, c=5) -> None:
        """a, b, and c are the three constants for the cost function"""
        self.a = a
        self.b = b
        self.c = c

    @staticmethod
    def calculate_area(box):
        """Calculate the area of a bounding box."""
        x, y, w, h = box
        return w * h
    
    @staticmethod
    def calculate_centroid(box):
        """Calculate the centroid of a bounding box."""
        x, y, w, h = box
        return (x + w / 2, y + h / 2)

    def cost(self, left_box, right_boxes):
        """Calculate the cost between a single left box and multiple right boxes."""
        left_centroid = np.array(object_matcher.calculate_centroid(left_box))
        right_centroids = np.array([object_matcher.calculate_centroid(box) for box in right_boxes])

        vert_diff = self.c * np.abs(left_centroid[1] - right_centroids[:, 1])
        horiz_diff = np.abs(left_centroid[0] - right_centroids[:, 0])
        horiz_diff[horiz_diff < 0] = self.b * np.abs(horiz_diff[horiz_diff < 0])
        area_diff = np.abs(object_matcher.calculate_area(left_box) - np.array([object_matcher.calculate_area(box) for box in right_boxes])) / self.a

        cost = vert_diff + horiz_diff + area_diff
        return cost

    def match(self, left_box, right_boxes):
        """Match a single left box with multiple right boxes."""
        cost = self.cost(left_box, right_boxes)
        best_match_index = np.argmin(cost)
        return best_match_index

if __name__ == "__main__":
    def main():
        ob = object_matcher()

        left_box = (10, 40, 20, 20)
        right_boxes = [(0, 38, 21, 20), (101, 38, 9, 18), (18, 95, 27, 9), (67, 68, 16, 21)]

        print("Left Box:", left_box)
        print("Right Boxes:", right_boxes)

        best_match_index = ob.match(left_box, right_boxes)

        print("Best Match Index:", best_match_index)

    main()