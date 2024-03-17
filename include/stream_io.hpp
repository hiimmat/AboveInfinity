#pragma once

#include <charconv>
#include <format>
#include <unordered_set>

#include "execute.hpp"

namespace ntensor {

/*
 * Special characters used while reading/writing a Tensor from/to a stream
 */
struct io_parameters {
  char delimiter{','};
  char newline{'\n'};
  char plane_start{'{'};
  char plane_end{'}'};
  char dimension_start{'['};
  char dimension_end{']'};
  char channels_start{'('};
  char channels_end{')'};
  char comment{'#'};
};

/*
 * Optional content that can be added to an output stream
 */
struct additional_output_content {
  std::string header;
  std::string footer;
};

inline namespace internal {

/*
 * Checks whether all IO parameters are unique
 * Parameters:
 * @tparam parameters: IO parameters that have to be validated
 * @return: true if all IO parameters are unique, false otherwise
 */
template <io_parameters parameters>
[[nodiscard]] consteval bool all_io_parameters_unique() {
  // Another approach would be to use a hash map
  std::array elements{parameters.delimiter,      parameters.newline,         parameters.plane_start,
                      parameters.plane_end,      parameters.dimension_start, parameters.dimension_end,
                      parameters.channels_start, parameters.channels_end,    parameters.comment};

  std::sort(elements.begin(), elements.end());

  for (std::size_t i = 1u; i < elements.size(); ++i) {
    if (elements[i] == elements[i - 1u]) {
      return false;
    }
  }

  return true;
}

/*
 * Converts a character sequence to an integer value
 * Parameters:
 * @param view: character sequence that is converted
 * @param value: variable in which the result of the conversion is stored
 * @return: true if the conversion succeeded, false otherwise
 */
[[nodiscard]] bool string_view_to_number(std::string_view view, auto& value) noexcept {
  if (view.empty()) {
    return false;
  }

  const char* first = view.data();
  const char* last = view.data() + view.length();

  std::from_chars_result res = std::from_chars(first, last, value);

  if (res.ec != std::errc()) {
    return false;
  }

  return res.ptr == last;
}

/*
 * Implementation of the write_to_sink method
 */
template <io_parameters parameters, typename Plane, typename Sink, typename Formatter>
void write_to_sink_impl(Plane&& plane, Sink&& sink, Formatter&& formatter) {
  using plane_type = std::decay_t<Plane>;
  static constexpr std::size_t rank = plane_type::rank();

  if constexpr (rank > 1u) {
    sink << parameters.dimension_start << parameters.newline;

    for (std::size_t i = 0u; i < plane_type::dimensions().template at<rank - 1U>(); ++i) {
      write_to_sink_impl<parameters>(plane.template like<remove_nth_element<rank - 1u>(plane_type::dimensions()),
                                                         remove_nth_element<rank - 1u>(plane_type::strides())>(
                                         i * plane_type::strides().template at<rank - 1u>() * plane_type::channels()),
                                     std::forward<Sink>(sink), std::forward<Formatter>(formatter));
    }
  } else if constexpr (rank == 1U) {
    static constexpr std::size_t dimension = plane_type::dimensions().template at<0u>();
    static constexpr std::size_t channels = plane_type::channels();

    sink << parameters.dimension_start;

    auto write_elements = [&](std::size_t d) {
      if constexpr (channels > 1u) {
        sink << parameters.channels_start;
      }

      formatter(sink, plane.at(d * plane_type::strides().template at<0u>() * plane_type::channels()));

      for (std::size_t c = 1u; c < plane_type::channels(); ++c) {
        sink << parameters.delimiter;
        formatter(sink, plane.at(d * plane_type::strides().template at<0u>() * plane_type::channels() + c));
      }

      if constexpr (channels > 1u) {
        sink << parameters.channels_end;
      }
    };

    write_elements(0u);

    for (std::size_t d = 1u; d < dimension; ++d) {
      sink << parameters.delimiter;
      write_elements(d);
    }
  }

  sink << parameters.dimension_end << parameters.newline;
}

/*
 * Implementation of the load_from_source method
 */
template <io_parameters parameters, typename Plane, typename Converter>
void load_from_source_impl(Plane&& plane, Converter&& converter, std::string_view& data) {
  std::size_t array_idx = 0u, channel = 0u;
  bool channels_closed = true;

#ifdef ENABLE_NT_EXPECTS
  int dimensions_open = 0;
  bool plane_closed = true;
#endif

  static constexpr char delimiters[]{parameters.delimiter,       parameters.newline,
                                     parameters.plane_start,     parameters.plane_end,
                                     parameters.dimension_start, parameters.dimension_end,
                                     parameters.channels_start,  parameters.channels_end,
                                     parameters.comment,         '\0'};

#ifdef ENABLE_NT_EXPECTS
  Expects(data.size());
#endif

  while (data.size()) {
    switch (data.front()) {
      case parameters.plane_start:
#ifdef ENABLE_NT_EXPECTS
        Expects(plane_closed);
        plane_closed = false;
#endif
        data.remove_prefix(1u);
        continue;
      case parameters.plane_end:
#ifdef ENABLE_NT_EXPECTS
        Expects(!plane_closed);
        plane_closed = true;
#endif
        data.remove_prefix(1u);
        goto expects;
      case parameters.dimension_start:
#ifdef ENABLE_NT_EXPECTS
        ++dimensions_open;
#endif
        data.remove_prefix(1u);
        continue;
      case parameters.dimension_end:
#ifdef ENABLE_NT_EXPECTS
        --dimensions_open;
#endif
        data.remove_prefix(1u);
        continue;
      case parameters.channels_start:
#ifdef ENABLE_NT_EXPECTS
        Expects(channels_closed);
#endif
        channels_closed = false;
        data.remove_prefix(1u);
        continue;
      case parameters.channels_end:
        ++array_idx;
#ifdef ENABLE_NT_EXPECTS
        Expects(!channels_closed);
#endif
        channels_closed = true;
        channel = 0u;
        data.remove_prefix(1u);
        continue;
      case parameters.delimiter:
      case parameters.newline:
        data.remove_prefix(1u);
        continue;
      case parameters.comment:
        // We've reached the footer
        goto expects;
    };

    auto string_value = data.substr(0u, data.find_first_of(delimiters));

    data.remove_prefix(string_value.size());

    using plane_type = typename std::decay_t<decltype(plane)>;

    const auto pos = compute_array_position_from_index<plane_type::channels()>(
        compute_unaligned_strides(plane_type::dimensions()), plane_type::strides(), channel, array_idx);
    typename plane_type::value_type value{};

    [[maybe_unused]] auto success = converter(string_value, value);

#ifdef ENABLE_NT_EXPECTS
    Expects(success);
#endif

    plane.at(pos) = value;
    !channels_closed ? ++channel : ++array_idx;
  }

expects:
#ifdef ENABLE_NT_EXPECTS
  Expects(dimensions_open == 0u);
  Expects(channels_closed);
  Expects(plane_closed);
#endif
  return;
};

}  // namespace internal

/*
 * Saves a tensor to a sink
 * Parameters:
 * @tparam parameters: Special characters used while writing the Tensor object
 * @param tensor: Tensor object that's going to be written to the given stream
 * @param sink: Stream to which the Tensor object is written
 * @param additional_content: Additional content that will be added to the stream, if it exists
 * @param formatter: User specified function used for custom formatting of the written tensor elements. For example, the
 * user can determine how real and imaginary numbers are stored in the stream, adjust the precision for floating-point
 * numbers, encode the values, store the values in binary format, etc.
 */
template <io_parameters parameters = {}, typename Tensor, typename Sink, typename Formatter>
void write_to_sink(Tensor&& tensor, Sink&& sink, const additional_output_content& additional_content,
                   Formatter&& formatter) {
  static_assert(all_io_parameters_unique<parameters>());

  auto write_additional_content = [&sink](auto& content, auto& comment, auto& newline) {
    if (!content.empty()) {
      sink << comment;

      for (auto c : content) {
        sink << c;

        if (c == '\n' || c == '\r') {
          sink << comment;
        }
      }
      sink << newline;
    }
  };

  write_additional_content(additional_content.header, parameters.comment, parameters.newline);

  std::string plane_delimiter{};

  for_each_plane(
      [&](auto&& plane) {
        static_assert((std::is_invocable_v<decltype(formatter), Sink, decltype(plane.at(0u))>));
        sink << plane_delimiter << parameters.plane_start << parameters.newline;
        write_to_sink_impl<parameters>(std::forward<decltype(plane)>(plane), sink, formatter);
        sink << parameters.plane_end;
        plane_delimiter = std::format("{}{}", parameters.delimiter, parameters.newline);
      },
      tensor);
  sink << parameters.newline << parameters.newline;

  write_additional_content(additional_content.footer, parameters.comment, parameters.newline);
}

/*
 * Adapter for the write_to_sink method that uses a predefined formatter
 * In this case no value formatting is applied. Instead the values are stored exactly as they are found in the Tensor
 * object Parameters:
 * @tparam parameters: Special characters used while writing the Tensor object
 * @param tensor: Tensor object that's going to be written to the given stream
 * @param sink: Stream to which the Tensor object is written
 * @param additional_content: Additional content that will be added to the stream, if it exists
 */
template <io_parameters parameters = {}, typename Tensor, typename Sink>
void write_to_sink(Tensor&& tensor, Sink&& sink, const additional_output_content& additional_content = {}) {
  write_to_sink<parameters>(tensor, sink, additional_content, [](auto& sink, const auto& value) { sink << value; });
}

/*
 * Reads a sequence of elements from a source into a Tensor object
 * Parameters:
 * @tparam parameters: Special characters used while reading the Tensor object
 * @param tensor: Tensor object to which the read elements are written to
 * @param data: string_view of the source from which the elements are read
 * @param converter: User specified function used for reading (parsing) the elements from the source
 */
template <io_parameters parameters = {}, typename Tensor, typename Converter>
void load_from_source(Tensor&& tensor, std::string_view data, Converter&& converter) {
  static_assert(all_io_parameters_unique<parameters>());

  // Skip header and empty lines
  while (data.starts_with(parameters.comment) || data.front() == parameters.newline) {
    data.remove_prefix(data.find_first_of(parameters.newline));
    data.remove_prefix(data.find_first_not_of(parameters.newline));
  }

  for_each_plane(
      [&converter, &data](auto& plane) {
        static_assert((std::is_invocable_v<Converter, std::string_view, decltype(plane.at(0u))>));
        load_from_source_impl<parameters>(plane, std::forward<Converter>(converter), data);
      },
      tensor);

  // TODO: Assure that there's no leftover data
}

/*
 * Adapter for the load_from_source method that uses a predefined converter
 * Warning
 * The default converter supports integer values only. For any other type, a custom converter needs to be supplied
 * Parameters:
 * @tparam parameters: Special characters used while reading the Tensor object
 * @param tensor: Tensor object to which the read elements are written to
 * @param data: string_view of the source from which the elements are read
 * TODO:
 * Replace string_view with a stream
 */
template <io_parameters parameters = {}, typename Tensor>
void load_from_source(Tensor&& tensor, std::string_view data) {
  load_from_source<parameters>(std::forward<Tensor>(tensor), data,
                               [](std::string_view view, auto& value) { return string_view_to_number(view, value); });
}

}  // namespace ntensor
