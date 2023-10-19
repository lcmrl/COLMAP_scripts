# colmap feature_extractor --database_path ./db.db --image_path ./ventimiglia_nadiral --camera_mode 0
# colmap exhaustive_matcher --database_path ./db.db
#colmap mapper --database_path ./db.db --image_path ./ventimiglia_nadiral --output_path ./outs1 --image_list_path ./part1.txt
#colmap mapper --database_path ./db.db --image_path ./ventimiglia_nadiral --output_path ./outs2 --image_list_path ./part2.txt
colmap model_merger --input_path1 ./outs1/0 --input_path2 ./outs2/0 --output_path ./joined