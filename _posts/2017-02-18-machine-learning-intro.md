---
title: "Introduction to Machine Learning"
categories: ML
date: 2017-02-18
tags: [machine learning, data science]
header:
  image: "/images/int1.jpg"
excerpt: "Data Science, Supervised Learning, Unsupervised Learning"
---




{% for post in site.categories.ML %}
 <li><span>{{ post.date | date_to_string }}</span> &nbsp; <a href="{{ post.url }}">{{ post.title }}</a></li>
{% endfor %}
