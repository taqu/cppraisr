#include <iostream>
#include <flags/include/flags.h>
#include "util.h"
#include "cppraisr.h"

void print_help()
{
}

int main(int argc, char** argv)
{
    using namespace cppraisr;
    const flags::args args(argc, argv);
    if(args.get<bool>("help", false)){
        print_help();
        return 0;
    }

    int32_t max_images = 2147483647;
    int32_t max_threads = -1;
    {
        std::optional<int32_t> option_images = args.get<int32_t>("max");
        if(option_images){
            max_images = option_images.value();
        }
        std::optional<int32_t> option_threads = args.get<int32_t>("threads");
        if(option_threads){
            max_threads = option_threads.value();
        }
        max_threads = std::max(max_threads, 1);
    }

    std::vector<std::filesystem::path> files = parse_directory("train_data",
                                                               [](const std::filesystem::directory_entry& entry) {
                                                                   std::filesystem::path extension = entry.path().extension();
                                                                   if(extension == ".jpg") {
                                                                       return true;
                                                                   }
                                                                   if(extension == ".png") {
                                                                       return true;
                                                                   }
                                                                   return false;
                                                               });
    std::unique_ptr<RAISRTrainer> trainer = std::make_unique<RAISRTrainer>();
    trainer->train(files, max_threads, max_images);
    return 0;
}
