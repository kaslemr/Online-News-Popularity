library(rmarkdown)

days_of_week = c("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

for (day in days_of_week){
  rmarkdown::render("online_news_automated_reports.Rmd", output_file = paste0(day,".md"), 
                    params = list(day_of_week = day))
}
