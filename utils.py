def calculate_price(units):
    """
    Calculate electricity price based on TNEB tariff details (Bi-monthly Cycle)

    Tariff structure:
    0–100 Units: Free
    101–200 Units: ₹2.35 – ₹4.80 per unit
    201–400 Units: ₹4.70 per unit
    401–500 Units: ₹6.30 per unit
    501–600 Units: ₹8.40 per unit
    601–800 Units: ₹9.45 per unit
    801–1000 Units: ₹10.50 per unit
    Above 1000 Units: ₹11.55 per unit

    Args:
        units (float): Total electricity consumption in kWh

    Returns:
        float: Total price in ₹
    """
    if units <= 100:
        return 0.0

    price = 0.0

    # 101-200 units: Use average rate of ₹3.575 per unit
    if units > 100:
        units_in_slab = min(units - 100, 100)
        price += units_in_slab * 3.575  # Average of ₹2.35 and ₹4.80

    # 201-400 units
    if units > 200:
        units_in_slab = min(units - 200, 200)
        price += units_in_slab * 4.70

    # 401-500 units
    if units > 400:
        units_in_slab = min(units - 400, 100)
        price += units_in_slab * 6.30

    # 501-600 units
    if units > 500:
        units_in_slab = min(units - 500, 100)
        price += units_in_slab * 8.40

    # 601-800 units
    if units > 600:
        units_in_slab = min(units - 600, 200)
        price += units_in_slab * 9.45

    # 801-1000 units
    if units > 800:
        units_in_slab = min(units - 800, 200)
        price += units_in_slab * 10.50

    # Above 1000 units
    if units > 1000:
        units_in_slab = units - 1000
        price += units_in_slab * 11.55

    return round(price, 2)