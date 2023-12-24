---
layout: page
permalink: /works/
title: research & projects
description: 
nav: true
nav_order: 1
---
<!-- _bibliography/papers -->
<div class="publications">
{% bibliography -f papers -q @*[displaytype=paper]* %}
</div>

<!-- _bibliography/preprints -->
<h1> preprints </h1>
<div class="publications">
{% bibliography -f papers -q @*[displaytype=preprint]* %}
</div>