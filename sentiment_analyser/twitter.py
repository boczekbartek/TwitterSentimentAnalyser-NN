import twitter
import logging


class TwitterDownloader:
    def __init__(self, credentials):
        self.api = twitter.Api(**credentials, sleep_on_rate_limit=True)
        self.data = []

    @staticmethod
    def extract_tweet_data(tweet):
        return {
            "created_at": tweet.created_at,
            "created_at_seconds": tweet.created_at_in_seconds,
            "id": tweet.id_str,
            "lang": tweet.lang,
            "user": tweet.user.name,
            "full_text": tweet.full_text
        }

    def tweets_with_hashtags_generator(self, hashtags, lang, max_tweets=1000):
        """
        Download tweets with given hashtags. Minimum is 100.
        :param hashtags: list of lists, if you want tweets with multiple hashtags inside or string (or 1 elem list) if
                         only one hashtag in tweet.
        """
        downloaded_tweets = 0
        previous_downloaded = -1
        max_id = 0

        while downloaded_tweets < max_tweets:
            if previous_downloaded == downloaded_tweets:
                print("No more tweets to download, ending...")
                break

            previous_downloaded = downloaded_tweets
            for in_tweet_hashtags in hashtags:

                if isinstance(in_tweet_hashtags, str):
                    hashtags_query = "%23" + in_tweet_hashtags
                elif isinstance(in_tweet_hashtags, list) or isinstance(in_tweet_hashtags, tuple):
                    hashtags_query = "%23" + in_tweet_hashtags[0] if len(
                        in_tweet_hashtags) == 1 else "%23" + "%23".join(in_tweet_hashtags)
                else:
                    raise (
                        TypeError(
                            "Elements of hashtags list have to be list or tuple - if multiple hashtags or string if 1 wanted"
                        )
                    )

                # This is next chunk of tweets so we want to download tweets older than the oldest already downloaded
                # Initialized by 0
                if max_id == 0:
                    query = 'q={}&lang={}&tweet_mode=extended&count=100'.format(hashtags_query, lang)
                else:
                    query = 'q={}&lang={}&tweet_mode=extended&count=100&max_id={}'.format(hashtags_query, lang,
                                                                                          max_id - 1)

                logging.info("Query: {}".format(hashtags_query))

                tweets = self.api.GetSearch(
                    raw_query=query
                )
                if len(tweets) == 0:
                    print("No more tweets for hashtag:", in_tweet_hashtags, sep=" ")
                    break

                for tweet in tweets:
                    parsed_tweet = self.extract_tweet_data(tweet)
                    self.data.append(parsed_tweet)
                    if max_id == 0:
                        max_id = parsed_tweet["id"]
                    else:
                        max_id = min(max_id, parsed_tweet["id"])
                    yield parsed_tweet
                    downloaded_tweets += 1
