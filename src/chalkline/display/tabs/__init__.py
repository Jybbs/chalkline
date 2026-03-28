"""
Tab rendering functions for the Chalkline Marimo dashboard.

Usage::

    from chalkline.display import tabs

    tabs.splash(logo_dir, metrics)
    tabs.your_match(ctx)
    tabs.career_paths(ctx, target_data, dropdown)
"""

from .context import TabContext

from .career_paths.render    import career_paths_tab    as career_paths
from .job_postings.render    import job_postings_tab    as job_postings
from .ml_internals.render    import ml_internals_tab    as ml_internals
from .next_steps.render      import next_steps_tab      as next_steps
from .resume_feedback.render import resume_feedback_tab as resume_feedback
from .splash.render          import splash_page         as splash
from .your_match.render      import your_match_tab      as your_match
