---
layout: post
title: Visualizing 2017 MRT Performance using D3
date: 2018-01-19 13:32:20 +0300
description: This is my entry towards accenture visualization challenge and its my first time using javascript and d3. # Add post description (optional)
img: train_thumbnail.jpg # Add image post (optional)
tags: [Holidays, Hawaii]
---

Philippines 2017 MRT Foot traffic Visualization made using D3. This was my entry for the Data visualization category for accenture's big data challenge.
This was also my first time using D3 and javascipt in general. And its also my first shot at data visualization. I try to summarize in this post the process I have taken to build my first ever visualization using d3 and my shortcoming in what I could have done better. I definetly learned a lot, during the process and I think I can do much better next time a challenge like this comes around.

## Gathering the Data

So before any visualization is to occur, I needed to first gather the necessary data. Open data philippines sadly only gives mrt data which is 2015 and below. So I opted to requesting data from eFOI Philippines and I asked for all data from 2015 till 2017. After 2 weeks I got the data, but what a fucking surpries, its in excel format. The excel file was configured to their own style, and its was terrifying. Nonetheless after some work I managed to get the mrt data into a json format, additional data such as MRT issues, I managed to scrape them off MRT website. Followed by some pandas manipulation for data cleaning and handling missing data, I was finally able to get them all in a nice csv format.

## Building my first D3 Graphs

>Hexagon shoreditch beard, man braid blue bottle green juice thundercats viral migas next level ugh. Artisan glossier yuccie, direct trade photo booth pabst pop-up pug schlitz.

I have never done javascript in my life, so I figured this is a good chance to have a go at it. The main reason I wanted to try D3, is just because people said it was kind of hard to learn(I am a masochist, lets GO). It took me a couple of days to get acclimated to javaxcript, but its a lot like python so I guess thats good.

When It came to the visualizing the data, my first dumbass thought is how could I present as much data as possible into an interactive chart, that would also serve as a seemless UI for anyone to use.

Key points I want to Visualize

* Project total entries and exits using a calendar heatmap z
* Turn the big calendar heatmap into interactive buttons
* The interactive buttons will display barcharts that can show data on each station
* Then add extra buttons to toggle interesting events such as mrt issues/breakdowns or Holidays

After compiling what I want to visualize, I followed a typical color scheme to try and make it look passable. When visualizing the traffic throughout the year, I used a calendar heatmap to give a view of the total foot traffic. I dont know why, but during the time I thought it would be usefull to add a side by side bar chart, so that when you click on the calendar heatmap it can give you a breakdown of amount of people entering and exiting a station. Hovering over the elements in the heatmap also gives additional details displayed by a tooltip.



![colorscheme]({{site.baseurl}}/assets/img/train_calendar_heatmap_2.png)



## Shortcoming and Learnings
