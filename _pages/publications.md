---
layout: page
permalink: /publications/
title: publications
description: 
nav: true
nav_order: 1
---
<!-- _pages/publications.md -->
<div class="publications">
{% bibliography -f papers {{ site.scholar.bibliography }} %}
</div>

<div class="preprints">
{% bibliography -f preprints {{ site.scholar.bibliography }} %}
</div>