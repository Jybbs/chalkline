"""
HTML card builders for the career report dashboard.

Each function returns a `mo.Html` object styled with `.cl-card` CSS
classes defined in `app/chalkline.css`. Cards adapt to dark and light
modes automatically through Marimo design token variables.
"""

import marimo as mo

from chalkline.collection.schemas import Posting


def _card(content: str) -> mo.Html:
    """
    Wrap inner HTML in a `.cl-card` container.
    """
    return mo.Html(f"<div class='cl-card'>{content}</div>")


def apprenticeship_card(
    min_hours   : int,
    rapids_code : str,
    trade       : str
) -> mo.Html:
    """
    Registered apprenticeship card with trade, RAPIDS code, and
    hour requirement.

    Args:
        min_hours   : Minimum required training hours.
        rapids_code : DOL RAPIDS apprenticeship identifier.
        trade       : Apprenticeship trade title.

    Returns:
        Styled card element.
    """
    return _card(
        f"<strong>{trade}</strong><br>"
        f"<span class='secondary'>RAPIDS {rapids_code}</span>"
        f" &middot; <span>{min_hours:,} hours minimum</span>"
    )


def board_card(
    best_for : str,
    category : str,
    focus    : str,
    name     : str
) -> mo.Html:
    """
    Job board card with name, focus area, best-for description,
    and category tag.

    Args:
        best_for : Description of who benefits most from this board.
        category : Board classification (e.g., "General", "Trade").
        focus    : Primary subject area of the board.
        name     : Display name of the job board.

    Returns:
        Styled card element.
    """
    return _card(
        f"<strong>{name}</strong>"
        f"<span class='badge'>{category}</span><br>"
        f"<span class='secondary'>{focus}</span><br>"
        f"<span class='meta'>{best_for}</span>"
    )


def card_grid(cards: list[mo.Html]) -> mo.Html:
    """
    Arrange cards in a responsive two-column grid.

    Wraps the provided card elements in a `.cl-card-grid` container
    that collapses to a single column on narrow viewports.

    Args:
        cards: Card elements to arrange.

    Returns:
        Grid container wrapping all cards.
    """
    return mo.Html(
        f"<div class='cl-card-grid'>{''.join(card.text for card in cards)}</div>"
    )


def employer_card(
    career_url  : str,
    member_type : str,
    name        : str,
    posting_url : str
) -> mo.Html:
    """
    AGC employer card with company name, member type, and links to
    the original posting and career page.

    Args:
        career_url  : URL to the employer's career page (empty string if
                      unavailable).
        member_type : AGC membership classification.
        name        : Company display name.
        posting_url : URL to the matched job posting.

    Returns:
        Styled card element.
    """
    return _card(
        f"<strong>{name}</strong>"
        f"<span class='badge'>{member_type}</span><br>"
        f"<a href='{posting_url}'>View Posting</a>"
        f"{f" &middot; <a href='{career_url}'>Career Page</a>" if career_url else ""}"
    )


def posting_card(posting: Posting) -> mo.Html:
    """
    Job posting card with title, company, location, date, truncated
    description, and a link to the original listing.

    Truncates the description at 200 characters to keep cards compact
    within the grid layout.

    Args:
        posting: Corpus posting record.

    Returns:
        Styled card element.
    """
    if len(description := posting.description) > 200:
        description = description[:200].rsplit(" ", 1)[0] + "..."

    meta = " &middot; ".join(filter(None, [
        posting.location or "Maine",
        posting.date_posted.strftime("%b %d, %Y") if posting.date_posted else None
    ]))

    return _card(
        f"<strong>{posting.title}</strong><br>"
        f"<span class='secondary'>{posting.company}</span><br>"
        f"<span class='meta'>{meta}</span>"
        f"<p>{description}</p>"
        f"<a href='{posting.source_url}'>View Posting</a>"
    )


def program_card(
    credential  : str,
    institution : str,
    name        : str,
    url         : str
) -> mo.Html:
    """
    Educational program card with program name, institution,
    credential type, and enrollment link.

    Args:
        credential  : Credential awarded (e.g., "AAS", "Certificate").
        institution : Name of the college or university.
        name        : Program title.
        url         : Link to the program page.

    Returns:
        Styled card element.
    """
    return _card(
        f"<strong>{name}</strong><br>"
        f"<span class='secondary'>{institution}</span>"
        f" &middot; <span>{credential}</span><br>"
        f"<a href='{url}'>Program Details</a>"
    )
