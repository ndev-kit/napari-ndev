import itertools
import string

import pandas as pd


class PlateMapper:
    """
    A class for creating and manipulating plate maps.

    Attributes
    ----------
    plate_size : int
        The size of the plate (e.g., 96, 384).
        Defaults to 96.
    leading_zeroes : bool
        Whether to include leading zeroes in the column labels.
        Defaults to False.
    treatments : dict
        A dictionary mapping treatments to conditions and well ranges.
    wells : dict
        A dictionary mapping plate sizes to the number of rows and columns.
    plate_map : pandas.DataFrame
        The plate map DataFrame with well labels.
    pivoted_plate_map : pandas.DataFrame
        The wide-formatted plate map DataFrame with treatments as columns.
        Pivots only one treatment at a time.
    styled_plate_map : pandas.io.formats.style.Styler
        The styled pivoted plate map DataFrame with different background
        colors for each unique value.

    Methods
    -------
    __init__(plate_size=96)
        Initializes a PlateMapper object.
    create_empty_plate_map()
        Creates an empty plate map DataFrame for a given plate size.
    assign_treatments(treatments)
        Assigns treatments to specific wells in a plate map.
    get_pivoted_plate_map(treatment)
        Pivots a plate map DataFrame to create a wide-formatted DataFrame with a single treatment as columns.
    get_styled_plate_map(treatment, palette='colorblind')
        Styles a plate map DataFrame with different background colors for each unique value.

    """

    def __init__(self, plate_size=96, treatments=None, leading_zeroes=False, ):
        """
        Initialize a PlateMapper object.

        Parameters
        ----------
        plate_size : int, optional
            The size of the plate. Defaults to 96.
        leading_zeroes : bool, optional
            Whether to include leading zeroes in the column labels.
            Defaults to False.
        treatments : dict, optional
            A dictionary mapping treatments to conditions and well ranges.
            If provided, the treatments will be assigned to the plate map.
            Defaults to None.

        """
        self.plate_size = plate_size
        self.leading_zeroes = leading_zeroes
        self.wells = {
            6: (2, 3),
            12: (3, 4),
            24: (4, 6),
            48: (6, 8),
            96: (8, 12),
            384: (16, 24),
        }
        self.plate_map = self.create_empty_plate_map()
        self.pivoted_plate_map = None
        self.styled_plate_map = None

        if treatments:
            self.assign_treatments(treatments)
            # pivot the first key in treatments
            self.get_styled_plate_map(next(iter(treatments.keys())))

    def create_empty_plate_map(self):
        """
        Create an empty plate map DataFrame for a given plate size.

        Returns
        -------
        pandas.DataFrame
            The empty plate map DataFrame with well labels.

        """
        num_rows, num_columns = self.wells[self.plate_size]

        row_labels = list(string.ascii_uppercase)[:num_rows]
        if self.leading_zeroes:
            column_labels = [f'{i:02d}' for i in range(1, num_columns + 1)]
        else:
            column_labels = list(range(1, num_columns + 1))

        well_rows = row_labels * num_columns
        well_rows.sort()  # needed to sort the rows correctly
        well_columns = column_labels * num_rows

        well_labels_dict = {'row': well_rows, 'column': well_columns}

        plate_map_df = pd.DataFrame(well_labels_dict)

        plate_map_df['well_id'] = plate_map_df['row'] + plate_map_df[
            'column'
        ].astype(str)
        self.plate_map = plate_map_df
        return plate_map_df

    def assign_treatments(self, treatments):
        """
        Assign treatments to specific wells in a plate map.

        Parameters
        ----------
        treatments : dict
            A dictionary mapping treatments to conditions and well ranges.

        Returns
        -------
        pandas.DataFrame
            The updated plate map with treatments assigned to specific wells.

        """
        for treatment, conditions in treatments.items():
            for condition, wells in conditions.items():
                for well in wells:
                    if ':' in well:
                        start, end = well.split(':')
                        start_row, start_col = start[0], int(start[1:])
                        end_row, end_col = end[0], int(end[1:])
                        well_condition = (
                            (self.plate_map['row'] >= start_row)
                            & (self.plate_map['row'] <= end_row)
                            & (self.plate_map['column'].astype(int) >= start_col)
                            & (self.plate_map['column'].astype(int) <= end_col)
                        )
                    else:
                        row, col = well[0], int(well[1:])
                        well_condition = (
                            (self.plate_map['row'] == row)
                            & (self.plate_map['column'] == col)
                        )

                    self.plate_map.loc[well_condition, treatment] = condition
        return self.plate_map

    def get_pivoted_plate_map(self, treatment):
        """
        Pivot a plate map DataFrame to create a wide-formatted DataFrame with a single treatment as columns.

        Parameters
        ----------
        treatment : str
            The column name of the treatment variable in the plate map DataFrame.

        Returns
        -------
        pandas.DataFrame
            The wide-formatted plate map DataFrame with treatments as columns.

        """
        plate_map_pivot = self.plate_map.pivot(
            index='row', columns='column', values=treatment
        )
        self.pivoted_plate_map = plate_map_pivot
        return plate_map_pivot

    def get_styled_plate_map(self, treatment, palette='tab10'):
        """
        Style a plate map with background colors for each unique value.

        Parameters
        ----------
        treatment : str
            The column name of the treatment variable in the plate map DataFrame.
        palette : str or list, optional
            The qualitative matplotlib colormap name to use for styling.
            Defaults to 'tab10'.
            Can be any qualitative matplotlib colormap name
            (e.g., 'tab10', 'tab20', 'Set1', 'Set2', 'Set3').

        Returns
        -------
        pandas.io.formats.style.Styler
            The styled plate map DataFrame with different background colors for each unique value.

        """
        from matplotlib import colormaps
        from matplotlib.colors import to_hex

        self.pivoted_plate_map = self.get_pivoted_plate_map(treatment)

        unique_values = pd.unique(self.pivoted_plate_map.values.flatten())
        unique_values = unique_values[pd.notna(unique_values)]

        cmap = colormaps[palette]
        # Get discrete colors from the colormap (works for qualitative palettes)
        colors_hex = [to_hex(cmap(i)) for i in range(cmap.N)]

        # Create an infinite iterator that cycles through the palette
        palette_cycle = itertools.cycle(colors_hex)
        color_dict = {value: next(palette_cycle) for value in unique_values}

        def get_background_color(value): # pragma: no cover
            if pd.isna(value):
                return ''
            return f'background-color: {color_dict[value]}'

        plate_map_styled = (
            self.pivoted_plate_map.style.applymap(get_background_color)
            .set_caption(f'{treatment} Plate Map')
            .format(lambda x: '' if pd.isna(x) else x)
        )
        self.styled_plate_map = plate_map_styled

        return plate_map_styled
