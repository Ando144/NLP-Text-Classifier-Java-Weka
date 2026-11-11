import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleStreamFilter;

import java.util.regex.Pattern;

/**
 * Custom Weka filter that normalizes email text.
 *
 *
 * This class processes the text attributes before
 * {@link weka.filters.unsupervised.attribute.StringToWordVector}
 * is applied, to reduce noise and obtain a better
 * vectorization.
 *
 *
 *
 * Main normalizations applied:
 *
 * <ul>
 * <li>URLs are turned into a single token</li>
 * <li>Email addresses are turned into a single token</li>
 * <li>Spaced-out letters are joined (e.g. "h e l l o" → "hello")</li>
 * <li>HTML attributes and fragments are removed</li>
 * <li>Repeated punctuation is simplified</li>
 * <li>Multiple spaces are normalized</li>
 * </ul>
 *
 *
 * The goal is to simplify the text so that the model learns
 * from patterns, not from noise.
 *
 *
 * <p>
 * Note: the class name must not change, because an instance of this
 * class is serialized in {@code model/normalizer.ser}.
 * </p>
 *
 * @version 1.0
 */
public class TextNormalizer extends SimpleStreamFilter {

    private static final long serialVersionUID = 1L;

    // Patterns: URL, email, spaced-out letters, HTML and punctuation
    private static final Pattern URL_PATTERN = Pattern.compile(
            "https?\\s*:\\s*/\\s*/\\s*\\S+|ftp\\s*:\\s*/\\s*/\\s*\\S+",
            Pattern.CASE_INSENSITIVE);

    private static final Pattern EMAIL_PATTERN = Pattern.compile(
            "[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}");

    private static final Pattern SPACED_LETTERS_PATTERN = Pattern.compile(
            "\\b([a-z])([\\s.]+[a-z]){3,}\\b");

    private static final Pattern HTML_ATTR_PATTERN = Pattern.compile(
            "\\b[a-z]+=[a-zA-Z0-9#]+\\b",
            Pattern.CASE_INSENSITIVE);

    private static final Pattern HTML_FRAGMENT_PATTERN = Pattern.compile(
            "\\b(tr|td|br|th|li|ul|ol|div|span|img|href|src|nbsp|bgcolor|cellpad){2,}\\b",
            Pattern.CASE_INSENSITIVE);

    private static final Pattern REPEATED_PUNCT_PATTERN = Pattern.compile(
            "([^a-zA-Z0-9\\s])\\1{2,}");

    private static final Pattern MULTI_SPACE_PATTERN = Pattern.compile(
            "\\s{2,}");

    @Override
    public String globalInfo() {
        return "Filter that normalizes the text of emails.";
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enableAllAttributes();
        result.enableAllClasses();
        result.enable(Capability.NO_CLASS);
        return result;
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) {
        return new Instances(inputFormat, 0);
    }

    // Process a single instance and normalize its text
    @Override
    protected Instance process(Instance instance) {
        Instance result = (Instance) instance.copy();

        for (int i = 0; i < instance.numAttributes(); i++) {
            if (instance.attribute(i).type() == Attribute.STRING
                    && !instance.isMissing(i)) {

                String text = instance.stringValue(i);
                text = normalize(text);

                result.setValue(i,
                        result.dataset().attribute(i).addStringValue(text));
            }
        }
        return result;
    }

    /**
     * Applies the normalization operations to a text string.
     *
     *
     * The transformations are applied in a fixed order
     * to guarantee a consistent result.
     *
     *
     * @param text Text to normalize
     * @return Normalized text
     */
    public static String normalize(String text) {
        if (text == null || text.isEmpty())
            return text;

        text = URL_PATTERN.matcher(text).replaceAll(" URLTOKEN ");
        text = EMAIL_PATTERN.matcher(text).replaceAll(" EMAILTOKEN ");
        text = collapseSpacedLetters(text);
        text = HTML_ATTR_PATTERN.matcher(text).replaceAll(" ");
        text = HTML_FRAGMENT_PATTERN.matcher(text).replaceAll(" ");
        text = REPEATED_PUNCT_PATTERN.matcher(text).replaceAll("$1");
        text = MULTI_SPACE_PATTERN.matcher(text).replaceAll(" ").trim();

        return text;
    }

    /**
     * Joins sequences of spaced-out single letters.
     *
     * <p>
     * For example: "h e l l o" → "hello"
     * </p>
     *
     * @param text Input text
     * @return Transformed text
     */
    private static String collapseSpacedLetters(String text) {
        StringBuffer sb = new StringBuffer();
        java.util.regex.Matcher m = SPACED_LETTERS_PATTERN.matcher(text);
        while (m.find()) {
            String matched = m.group();
            String collapsed = matched.replaceAll("[\\s.]+", "");
            m.appendReplacement(sb, collapsed);
        }
        m.appendTail(sb);
        return sb.toString();
    }
}
