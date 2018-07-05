var NewsFeed = function () {

    var feed = {
        newsItems: [],
        activeNewsItem: 0,

        update: function () {
            var newsItem = this.newsItems[this.activeNewsItem];
            if (this.activeNewsItem < 1) $("#news-feed .news-prev").css("opacity", "0");
            if (this.activeNewsItem > 1) $("#news-feed .news-next").css("opacity", "0");
            $("#news-feed .news-date").text(newsItem.date);
            $("#news-feed .news-headline").html(this.breakIntoSpans(newsItem.headline));
            $("#news-feed .news-more-href").attr("href", newsItem.href);
            this.addEllipses();
        },

        advance: function () {
            if (this.activeNewsItem < 2) {
                this.activeNewsItem += 1;
                this.update();
                $("#news-feed .news-prev").css("opacity", "1");
            }
        },

        previous: function () {
            if (this.activeNewsItem > 0) {
                this.activeNewsItem -= 1;
                this.update();
                $("#news-feed .news-next").css("opacity", "1");
            }
        },

        breakIntoSpans: function (headline) {
            var spans = headline.split(" ").map(function (word) {
                    return "<span>" + word + "</span>";
            });
            return spans.join(" ");
        },

        addEllipses: function () {
            var tooBig = false;
            $(".news-headline span").each(function () {
                if (tooBig) {
                    $(this).text("");
                }
                else if ($(this).next().position() && $(this).next().position().top > 15) {
                    tooBig = true;
                    $(this).text("...");
                }
            });
        },

        init: function () {
            var self = this;
            $.get("news.html", function (data) {
                var doc = $("<div/>");
                doc.html(data);

                var news = doc.find(".news:first");
                var articles = news.find(".fp-section-wide");
                var firstThreeArticles = articles.filter(function (index) {
                    return index < 3;
                });

                self.newsItems = firstThreeArticles.map(function () {
                    return {
                        date: $(this).find(".news-date").text(),
                        headline: $(this).find(".headline").text(),
                        href: "news#" + $(this).attr("id")
                    };
                });

            }).done(function () {
                self.update();
            });
        }
    };

    return feed;
};

$(document).ready(function () {
    feed = NewsFeed();
    feed.init();

    $(".news-next").click(function () {
        feed.advance();
    });

    $(".news-prev").click(function () {
        feed.previous();
    });
});

