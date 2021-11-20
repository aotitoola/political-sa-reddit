# from pmaw import PushshiftAPI
from psaw import PushshiftAPI
from datetime import date, datetime
import pandas as pd


# increasing the score improved the quality of the data
CURRENT_SCORE_THRESHOLD = 2


def date_to_timestamp(year, month, day):
    dt = datetime(year=year, month=month, day=day)
    timestamp = int(dt.timestamp())
    return timestamp


def filter_fxn(item):
    return item['score'] >= CURRENT_SCORE_THRESHOLD


def run_script(subreddit, fetch_type='comments', limit=1000, after=None, before=None):

    if not after:
        after = [2020, 1, 1]

    if not before:
        before = [2021, 1, 1]

    api = PushshiftAPI(shards_down_behavior=None)

    after_date = date_to_timestamp(*after)
    before_date = date_to_timestamp(*before)

    if fetch_type == 'comments':
        results = api.search_comments(subreddit=subreddit, limit=limit, filter_fn=filter_fxn, after=after_date,
                                      before=before_date)
    else:
        results = api.search_submissions(subreddit=subreddit, limit=limit, filter_fn=filter_fxn, after=after_date,
                                         before=before_date)

    # mem_safe = True, safe_exit = True
    results_df = pd.DataFrame(results)
    return results_df


def run_psaw(subreddit, fetch_type='comments', limit=1000, after=None, before=None):
    if not after:
        after = [2020, 1, 1]

    if not before:
        before = [2021, 1, 1]

    api = PushshiftAPI()

    after_date = date_to_timestamp(*after)
    before_date = date_to_timestamp(*before)

    if fetch_type == 'comments':
        gen = api.search_comments(after=after_date,
                                  subreddit=subreddit,
                                  filter=['body', 'score', 'subreddit'],
                                  limit=limit)
    else:
        gen = api.search_submissions(after=after_date,
                                     subreddit=subreddit,
                                     # filter=['url','author', 'title', 'subreddit'],
                                     limit=limit)

    res_df = pd.DataFrame([x.d_ for x in gen])
    res_df = res_df[['body', 'score', 'subreddit']]

    # remove all entries where comments/submissions have been deleted/removed
    res_df = res_df[res_df['body'] != '[removed]']
    res_df = res_df[res_df['body'] != 'deleted']

    return res_df


def fetch_reddit_comments(subreddit):
    print(f'Retrieving comments from /r/{subreddit}')
    # results = run_script(subreddit=subreddit, fetch_type='comments', limit=300, after=[2016, 1, 1], before=[2021, 1, 1])
    results = run_psaw(subreddit=subreddit, fetch_type='comments', limit=300, after=[2016, 1, 1], before=[2021, 1, 1])
    print(f'Retrieved {len(results)} comments from Pushshift\n')
    return results
