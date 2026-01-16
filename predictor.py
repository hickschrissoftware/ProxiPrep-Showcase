"""
Predictor module for ProxiPrep.

This module analyzes current inventory levels against PAR levels
to generate shopping lists for restocking.
"""

import pandas as pd


class PredictorEngine:
    """
    Analyzes inventory needs based on PAR levels.

    Compares current inventory quantities against predefined PAR levels
    to determine what items need to be restocked.
    """

    # Hardcoded PAR levels for common inventory items
    PAR_LEVELS = {
        'Tomatoes': 10,
        'Onions': 5,
        'Beer': 20,
        'Lettuce': 8,
        'Chicken': 15,
        'Beef': 12,
        'Cheese': 10,
        'Bread': 25,
        'Eggs': 30,
        'Milk': 10,
    }

    def __init__(self) -> None:
        """Initialize the PredictorEngine."""
        pass

    def analyze_needs(self, current_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare current inventory against PAR levels.

        Analyzes the current inventory DataFrame and identifies items
        that are below their PAR level threshold.

        Args:
            current_df: DataFrame with 'Item' and 'Quantity' columns
                       representing current inventory levels.

        Returns:
            A shopping_list DataFrame containing items where quantity
            is below PAR level, with columns:
            ['Item', 'Current_Qty', 'PAR_Level', 'Needed'].
        """
        if current_df.empty:
            return pd.DataFrame(columns=['Item', 'Current_Qty', 'PAR_Level', 'Needed'])

        shopping_list = []

        for _, row in current_df.iterrows():
            item_name = row['Item']
            current_qty = int(row['Quantity'])

            # Check if item has a defined PAR level
            if item_name in self.PAR_LEVELS:
                par_level = self.PAR_LEVELS[item_name]

                # Add to shopping list if below PAR
                if current_qty < par_level:
                    needed = par_level - current_qty
                    shopping_list.append({
                        'Item': item_name,
                        'Current_Qty': current_qty,
                        'PAR_Level': par_level,
                        'Needed': needed
                    })

        shopping_list_df = pd.DataFrame(shopping_list)

        if shopping_list_df.empty:
            shopping_list_df = pd.DataFrame(
                columns=['Item', 'Current_Qty', 'PAR_Level', 'Needed']
            )

        return shopping_list_df

    def get_par_level(self, item_name: str) -> int:
        """
        Get the PAR level for a specific item.

        Args:
            item_name: Name of the inventory item.

        Returns:
            The PAR level for the item, or 0 if not defined.
        """
        return self.PAR_LEVELS.get(item_name, 0)

    def set_par_level(self, item_name: str, level: int) -> None:
        """
        Set or update the PAR level for an item.

        Args:
            item_name: Name of the inventory item.
            level: The new PAR level to set.
        """
        self.PAR_LEVELS[item_name] = level
