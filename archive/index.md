---
layout: default
title: archive
---
<article class="page">
  <h1 class="page-title">blog archive</h1> <!-- {{ site.tagline }} --> 

{% for post in site.posts %}
  * {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{ post.url }})
{% endfor %}

</article>